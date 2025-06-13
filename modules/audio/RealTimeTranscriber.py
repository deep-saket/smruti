class RealTimeTranscriber:
    """
    Ties an AudioStreamer to an audio inference component for real-time transcription.
    """
    def __init__(self, infer_component):
        self.infer = infer_component

    def transcribe_stream(self, streamer):
        """
        Continuously pull audio blocks from the streamer and print transcriptions.
        """
        print("▶️  Starting real-time transcription (Ctrl+C to stop)...")
        try:
            for chunk in streamer.generator():
                text = self.infer.infer(chunk)
                print(f"📝 {text}")
        except KeyboardInterrupt:
            print("\n⏹️  Transcription stopped by user.")