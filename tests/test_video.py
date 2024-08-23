import os
import time

import cv2

import app.video.viz as viz
from app.video import PersonDetector

DATA_DIR = "../data"

video_capture = cv2.VideoCapture(0)

face_analyzer = PersonDetector(
    persons={
        "Max": os.path.join(DATA_DIR, "max.jpg"),
        "Artem": os.path.join(DATA_DIR, "artem.jpg"),
    }
)

predict_pause_number_frames = 5
k = 0

while True:
    start_time = time.time()
    ret, frame = video_capture.read()

    if k % predict_pause_number_frames == 0:
        predictions = face_analyzer(frame)
        if predictions is None:
            continue
    k += 1

    names = predictions["names"]
    face_locations = predictions["locations"]
    face_landmarks = predictions["landmarks"]

    for i in range(len(face_locations)):
        viz.draw_person_name(frame, names[i], face_locations[i])

    cv2.imshow("Video", frame)

    print(f"Time: {time.time() - start_time:.5f}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
