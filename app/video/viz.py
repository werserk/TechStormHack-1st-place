import numpy as np
from PIL import Image, ImageDraw, ImageFont

GREEN_COLOR = (0, 255, 0)
THICKNESS = 2
fontpath = "../data/font/Montserrat-Regular.ttf"
FONT = ImageFont.truetype(fontpath, 24)


def draw_person_name(image: np.array, name: str, coords: tuple) -> np.array:
    top, right, bottom, left = coords
    right_padding = 20
    right_line = 200
    top_padding = 20

    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)

    draw.line([(right, top), (right + right_padding, top - top_padding)], fill=GREEN_COLOR, width=THICKNESS)
    draw.line([(right + right_padding, top - top_padding), (right + right_padding + right_line, top - top_padding)],
              fill=GREEN_COLOR, width=THICKNESS)

    # Draw text
    text_position = (right + right_padding, top - top_padding * 3)
    draw.text(text_position, name, font=FONT, fill=GREEN_COLOR)
    return np.array(image_pil)


def draw_landmarks(image: np.array, landmarks: dict) -> np.array:
    image_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(image_pil)

    for name, points in landmarks.items():
        for point in points:
            draw.ellipse((point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1), fill=GREEN_COLOR)
    return np.array(image_pil)
