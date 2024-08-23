from app.audio.speaker_classifier import SpeakerClassifier
from app.audio.text_transcriber import TextTranscriber


class ProductionAudioAnalyzer:
    def __init__(self):
        self.speaker_classifier = SpeakerClassifier()
        self.text_transcriber = TextTranscriber()
