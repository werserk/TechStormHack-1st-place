import cv2
import numpy as np

GREEN_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_DUPLEX
THICKNESS = 2


def draw_person_name(frame: np.ndarray, name: str, coords: tuple) -> None:
    top, right, bottom, left = coords
    right_padding = 20
    right_line = 200
    top_padding = 20

    cv2.line(frame, (right, top), (right + right_padding, top - top_padding), GREEN_COLOR, THICKNESS)
    cv2.line(
        frame,
        (right + right_padding, top - top_padding),
        (right + right_line, top - top_padding),
        GREEN_COLOR,
        THICKNESS,
    )
    cv2.putText(frame, name, (right + right_padding, top - top_padding * 2), FONT, 1, GREEN_COLOR, THICKNESS)
