# modules/audio.py

import sounddevice as sd
import numpy as np
from queue import Queue

class AudioRecorder:
    """
    AudioRecorder captures audio from the default microphone.
    - samplerate: sampling rate in Hz (default 16000)
    - channels: number of audio channels (default 1)
    - dtype: data type for recording (default 'int16')
    """
    def __init__(self, samplerate=16000, channels=1, dtype='int16'):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype

    def record(self, duration):
        """
        Record audio from the microphone for the specified duration (in seconds).
        Returns a 1D NumPy array of normalized float32 samples.
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
        return audio

class AudioPlayer:
    """
    AudioPlayer plays back audio arrays through the default speaker.
    - samplerate: sampling rate in Hz (default 16000)
    """
    def __init__(self, samplerate=16000):
        self.samplerate = samplerate

    def play(self, audio_array):
        """
        Play a 1D NumPy audio array.
        """
        print("🔊 Playing back the recorded audio...")
        sd.play(audio_array, self.samplerate)
        sd.wait()

class AudioStreamer:
    """
    Continuously captures audio in fixed-size blocks.
    - samplerate: sampling rate in Hz.
    - channels: number of channels.
    - dtype: audio data type.
    - block_duration: length of each block in seconds.
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
        """
        while True:
            data = self.queue.get()
            audio = data.flatten().astype(np.float32) / np.iinfo(self.dtype).max
            yield audio
