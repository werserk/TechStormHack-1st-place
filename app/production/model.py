import os
import warnings
from typing import List, Dict

from app.audio import TextTranscriberOnline, SpeakerClassifier
from app.video import PersonDetector, main_loop
from app.video.detector import DATA_DIR

warnings.simplefilter("ignore")

DATA_DIR = "../data"


class ProductionModel:
    def __init__(self) -> None:
        self.text_transcriber = TextTranscriberOnline()
        self.speaker_classifier = SpeakerClassifier()
        self.person_detector = PersonDetector()

    def listen(self) -> None:
        self.text_transcriber.listen()

    def watch(self) -> None:
        main_loop(self.person_detector)

    def classify_speakers(self, audio_path: str) -> List[Dict[str, object]]:
        return self.speaker_classifier(audio_path)
