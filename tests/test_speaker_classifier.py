from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN_HERE",
)

diarization = pipeline("../data/audio/introduction.wav")

for segment, _, speaker in diarization.itertracks(yield_label=True):
    start_time = segment.start
    end_time = segment.end
    print(f"{start_time:.1f}-{end_time:.1f}s:{speaker}")
