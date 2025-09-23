import sounddevice as sd
import numpy as np
import webrtcvad
from queue import Queue
from models import ModelManager


class AudioRecorder:
    """
    AudioRecorder captures audio from the default microphone.

    Methods:
    - record(duration): Record a fixed-duration clip.
    - record_until_speech_end(frame_duration_ms=30, padding_duration_ms=300, vad_mode=1):
      Record until the user stops speaking (VAD-based).

    Attributes:
    - samplerate (int): Sampling rate in Hz.
    - channels (int): Number of audio channels.
    - dtype (str): Data type for recording (e.g., 'int16').
    """
    def __init__(self, samplerate=16000, channels=1, dtype='int16', speech_denoiser=None):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.speech_denoiser = speech_denoiser # if speech_denoiser else ModelManager.speech_denoiser

    def record(self, duration):
        """
        Record audio from the microphone for the specified duration.

        Args:
            duration (float): Length of recording in seconds.

        Returns:
            np.ndarray: 1D float32 array of normalized audio samples.
        """
        print(f"🎙️  Recording for {duration} seconds...")
        recording = sd.rec(
            int(duration * self.samplerate),
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype
        )
        sd.wait()
        audio = recording.flatten().astype(np.float32) / np.iinfo(self.dtype).max

        if self.speech_denoiser:
            audio = self.speech_denoiser.process(audio)

        return audio

    def listen(self,
               frame_duration_ms: int = 30,
               padding_duration_ms: int = 300,
               vad_mode: int = 1):
        """
        Continuously listen and yield utterances detected by VAD.

        This generator starts capturing audio immediately and uses voice
        activity detection (VAD) to find speech boundaries.  Each time the
        user speaks and then stops, the concatenated audio is yielded as a
        float32 array.  It then resumes listening for the next utterance.
        """
        while True:
            yield self.record_until_speech_end(frame_duration_ms,
                                               padding_duration_ms,
                                               vad_mode)

    def record_until_speech_end(self, frame_duration_ms=30, padding_duration_ms=300, vad_mode=1):
        """
        Record audio until the user stops speaking, using WebRTC VAD.

        This method continuously captures short frames and uses a
        Voice Activity Detector to detect speech onset and offset.

        Args:
            frame_duration_ms (int): Frame length for VAD in milliseconds (10, 20, or 30).
            padding_duration_ms (int): Amount of trailing silence (ms) before stopping.
            vad_mode (int): VAD aggressiveness (0-3, higher is less sensitive).

        Returns:
            np.ndarray: 1D float32 array of the captured utterance.
        """
        # Calculate sizes
        frame_size = int(self.samplerate * frame_duration_ms / 1000)
        padding_frames = int(padding_duration_ms / frame_duration_ms)

        # Prepare VAD and queue
        vad = webrtcvad.Vad(vad_mode)
        q = Queue()
        stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=frame_size,
            callback=lambda indata, frames, time, status: q.put(indata.copy())
        )
        stream.start()

        ring_buffer = []
        voiced_frames = []
        triggered = False
        silent_counter = 0

        while True:
            frame = q.get()
            pcm = frame.tobytes()
            is_speech = vad.is_speech(pcm, sample_rate=self.samplerate)

            if not triggered:
                if is_speech:
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
                else:
                    ring_buffer.append(frame)
                    if len(ring_buffer) > padding_frames:
                        ring_buffer.pop(0)
            else:
                voiced_frames.append(frame)
                if not is_speech:
                    silent_counter += 1
                    if silent_counter > padding_frames:
                        break
                else:
                    silent_counter = 0

        stream.stop()
        stream.close()

        # Concatenate, normalize, and return
        audio = np.concatenate(voiced_frames, axis=0)

        if self.speech_denoiser:
            t = ModelManager.stt.infer(audio).strip()
            audio = self.speech_denoiser.process(audio)

        return audio.flatten().astype(np.float32) / np.iinfo(self.dtype).max


class AudioPlayer:
    """
    AudioPlayer plays back audio arrays through the default speaker.

    Attributes:
    - samplerate (int): Sampling rate in Hz.
    """
    def __init__(self, samplerate=12000):
        self.samplerate = samplerate

    def play(self, audio_array):
        """
        Play a 1D NumPy audio array.

        Args:
            audio_array (np.ndarray): 1D array of audio samples (float or int).
        """
        print("🔊 Playing back the recorded audio...")
        sd.play(audio_array, self.samplerate)
        sd.wait()


class AudioStreamer:
    """
    Continuously captures audio in fixed-size blocks.

    Attributes:
    - samplerate (int): Sampling rate in Hz.
    - channels (int): Number of audio channels.
    - dtype (str): Audio data type.
    - block_duration (float): Duration of each block in seconds.
    """
    def __init__(self, samplerate=16000, channels=1, dtype='int16', block_duration=1.0):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.block_size = int(block_duration * samplerate)
        self.queue = Queue()
        self.stream = None

    def _callback(self, indata, frames, time, status):
        self.queue.put(indata.copy())

    def start(self):
        """
        Start the audio input stream.
        """
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.block_size,
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        """
        Stop and close the audio stream.
        """
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def generator(self):
        """
        Yield normalized float32 numpy arrays for each captured block.

        Yields:
            np.ndarray: 1D float32 audio block.
        """
        while True:
            data = self.queue.get()
            audio = data.flatten().astype(np.float32) / np.iinfo(self.dtype).max
            yield audio