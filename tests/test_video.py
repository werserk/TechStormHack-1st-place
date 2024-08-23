import os
import time

import cv2

import app.video.viz as viz
from app.video import FaceAnalytics

DATA_DIR = "../data"

video_capture = cv2.VideoCapture(0)

face_analyzer = FaceAnalytics(
    characters={
        "Max": os.path.join(DATA_DIR, "max.jpg"),
        "Artem": os.path.join(DATA_DIR, "artem.jpg"),
    }
)

while True:
    start_time = time.time()
    ret, frame = video_capture.read()
    predictions = face_analyzer(frame)
    if predictions is None:
        continue

    names = predictions["names"]
    face_locations = predictions["locations"]
    face_landmarks = predictions["landmarks"]

    for i in range(len(face_locations)):
        viz.draw_face_bbox(frame, names[i], face_locations[i])

    cv2.imshow("Video", frame)

    print(f"Time: {time.time() - start_time:.5f}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
