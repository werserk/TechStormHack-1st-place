import os

from app.video.detector import PersonDetector, main_loop

DATA_DIR = "../data"

person_detector = PersonDetector(
    persons={
        "Max": os.path.join(DATA_DIR, "max.jpg"),
        "Artem": os.path.join(DATA_DIR, "artem.jpg"),
    }
)

main_loop(person_detector)
