import warnings

from app.audio import TextTranscriber, SpeakerClassifier
from app.video import PersonDetector

warnings.simplefilter("ignore")


class ProductionModel:
    def __init__(self):
        self.text_transcriber = TextTranscriber()
        self.speaker_classifier = SpeakerClassifier()
        self.person_detector = PersonDetector()
        print("Production model initialized.")

    def __call__(self):
        pass


def main():
    ProductionModel()


if __name__ == '__main__':
    main()
