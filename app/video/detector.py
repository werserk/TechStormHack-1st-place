from typing import Dict, List, Optional

import cv2
import face_recognition
import numpy as np

from app.people.person import Person

DATA_DIR = "../data"


class PersonDetector:
    UNKNOWN_NAME = "unknown"

    def __init__(self, persons: Optional[List[Person]] = None):
        self.downscale_factor = 0.5
        self._persons = {}
        self._persons_names = []
        self._persons_encodings = []
        if persons is not None:
            self.persons = persons

    @property
    def persons(self):
        return self._persons

    @persons.setter
    def persons(self, persons: Optional[List[Person]] = None):
        if persons is None:
            persons = []
        self._persons = {
            person.name: {
                "path": person.image_path,
                "encoding": face_recognition.face_encodings(face_recognition.load_image_file(person.image_path))[0],
            }
            for person in persons
        }
        self._persons_names = list(self._persons.keys())
        self._persons_encodings = [self._persons[name]["encoding"] for name in self._persons_names]

    def find_person_faces(self, frame: np.ndarray, face_locations: List, threshold: float = 0.6):
        face_encodings = face_recognition.face_encodings(frame, known_face_locations=face_locations)
        predict_names = []

        for i in range(len(face_locations)):
            face_encoding = face_encodings[i]
            face_distances = face_recognition.face_distance(self._persons_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            name = self.UNKNOWN_NAME
            if face_distances[best_match_index] < threshold:
                name = self._persons_names[best_match_index]
            predict_names.append(name)
        return predict_names

    def downscale_image(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, None, fx=self.downscale_factor, fy=self.downscale_factor)

    def upscale_coords(self, coords: np.array) -> np.array:
        return (coords / self.downscale_factor).astype(np.int32)

    def __call__(self, frame: np.ndarray) -> Optional[Dict]:
        if frame is None:
            return None
        resized_frame = self.downscale_image(frame)
        resized_frame_rgb = np.ascontiguousarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

        face_locations = face_recognition.face_locations(resized_frame_rgb)
        names = self.find_person_faces(resized_frame_rgb, face_locations)

        face_landmarks = face_recognition.face_landmarks(resized_frame_rgb, face_locations)
        for landmark in face_landmarks:
            for name, points in landmark.items():
                landmark[name] = self.upscale_coords(np.array(points))

        return {
            "names": names,
            "locations": self.upscale_coords(np.array(face_locations)),
            "landmarks": face_landmarks,
        }
