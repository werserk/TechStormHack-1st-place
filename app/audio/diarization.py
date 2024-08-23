from pyannote.audio import Pipeline


class SpeakerClassifier:
    def __init__(self) -> None:
        self.model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="YOUR_HF_TOKEN_HERE")

    def __call__(self, audio_file: str) -> None:
        diarization = self.model("../data/audio/test.wav")
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start_time = segment.start
            end_time = segment.end
            print(f"Speaker {speaker} speaks from {start_time:.1f}s to {end_time:.1f}s")
