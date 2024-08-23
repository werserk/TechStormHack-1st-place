from typing import Dict, List, Optional

import cv2
import face_recognition
import numpy as np


class FaceAnalytics:
    def __init__(self, characters: Dict[str, str]):
        self.downscale_factor = 0.2
        self._characters = {}
        self._character_names = []
        self._character_encodings = []
        self.characters = characters

    @property
    def characters(self):
        return self._characters

    @characters.setter
    def characters(self, characters: Dict[str, str]):
        self._characters = {
            name: {
                "path": value,
                "encoding": face_recognition.face_encodings(
                    face_recognition.load_image_file(value)
                )[0],
            }
            for name, value in characters.items()
        }
        self._character_names = list(self._characters.keys())
        self._character_encodings = [
            self._characters[name]["encoding"] for name in self._character_names
        ]

    def find_character_faces(self, frame: np.ndarray, face_locations: List):
        face_encodings = face_recognition.face_encodings(
            frame, known_face_locations=face_locations
        )
        predict_names = []

        for i in range(len(face_locations)):
            face_encoding = face_encodings[i]
            matches = face_recognition.compare_faces(
                self._character_encodings, face_encoding, tolerance=0.6
            )
            face_distances = face_recognition.face_distance(
                self._character_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)

            name = "<unk>"
            if matches[best_match_index]:
                name = self._character_names[best_match_index]
            predict_names.append(name)
        return predict_names

    def downscale_image(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(
            frame, None, fx=self.downscale_factor, fy=self.downscale_factor
        )

    def upscale_coords(self, coords: np.array) -> np.array:
        return (coords / self.downscale_factor).astype(np.int32)

    def __call__(self, frame: np.ndarray) -> Optional[Dict]:
        if frame is None:
            return None
        resized_frame = self.downscale_image(frame)
        resized_frame_rgb = np.ascontiguousarray(
            cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        )

        face_locations = face_recognition.face_locations(resized_frame_rgb)
        names = self.find_character_faces(resized_frame_rgb, face_locations)

        face_landmarks = face_recognition.face_landmarks(
            resized_frame_rgb, face_locations
        )
        for landmark in face_landmarks:
            for name, points in landmark.items():
                landmark[name] = self.upscale_coords(np.array(points))

        return {
            "names": names,
            "locations": self.upscale_coords(np.array(face_locations)),
            "landmarks": face_landmarks,
        }
