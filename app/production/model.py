import os
import warnings

from app.audio import TextTranscriber, SpeakerClassifier
from app.video import PersonDetector, main_loop
from app.video.detector import DATA_DIR

warnings.simplefilter("ignore")

DATA_DIR = "../data"


class ProductionModel:
    def __init__(self):
        self.text_transcriber = TextTranscriber()
        self.speaker_classifier = SpeakerClassifier()
        self.person_detector = PersonDetector(
            persons={
                "Max": os.path.join(DATA_DIR, "max.jpg"),
                "Artem": os.path.join(DATA_DIR, "artem.jpg"),
            }
        )
        print("Production model initialized.")

    def listen(self):
        self.text_transcriber.listen()

    def watch(self):
        main_loop(self.person_detector)
