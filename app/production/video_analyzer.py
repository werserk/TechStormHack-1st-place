import logging
import os

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from tqdm import tqdm

import app.video.viz as viz
from app.audio.speaker_classifier import SpeakerClassifier
from app.people.person import Person
from app.video.detector import PersonDetector

fontpath = "../data/font/Montserrat-Regular.ttf"
FONT = ImageFont.truetype(fontpath, 12)

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

GREEN_COLOR = (0, 255, 0)

DATA_DIR = "../data"

persons_part1 = [
    Person("Александр", "Пушной", os.path.join(DATA_DIR, "Александр_Пушной.png")),
    Person("Алексей", "Вершинин", os.path.join(DATA_DIR, "Алексей_Вершинин.jpg")),
    Person("Андрей", "Ургант", os.path.join(DATA_DIR, "Андрей_Ургант.png")),
    Person("Дмитрий", "Колдун", os.path.join(DATA_DIR, "Дмитрий_Колдун.jpg")),
    Person("Евгений", "Папунаишвили", os.path.join(DATA_DIR, "Евгений_Папунаишвили.jpg")),
    Person("Евгений", "Рыбов", os.path.join(DATA_DIR, "Евгений_Рыбов.jpg")),
]

part2_dir = os.path.join(DATA_DIR, "part2")

persons_part2 = [
    Person("Джиган", "", os.path.join(part2_dir, "Джиган.png")),
    Person("Ведущий", "", os.path.join(part2_dir, "Ведущий.png")),
    Person("М1", "", os.path.join(part2_dir, "М1.png")),
    Person("Леди1", "", os.path.join(part2_dir, "Леди1.png")),
    Person("Пушной", "", os.path.join(part2_dir, "Пушной.png")),
    Person("Леди2", "", os.path.join(part2_dir, "Леди2.png")),
]


class VideoAnalyzer:
    def __init__(self) -> None:
        persons = persons_part2
        self.persons = {person.name: person for person in persons}
        self.speaker_classifier = SpeakerClassifier()
        self.person_detector = PersonDetector(persons=persons)

    @staticmethod
    def convert_video_to_audio(video_path: str, temp_audio_path: str) -> None:
        """Конвертирует видеофайл в аудиофайл."""
        logger.info(f"Converting video to audio...")
        audio = AudioSegment.from_file(video_path, format=video_path.split(".")[-1])
        audio.export(temp_audio_path, format="wav")

    def analyze_speakers(self, audio_path: str):
        """Анализирует аудио и возвращает фразы со временем и спикерами."""
        logger.info(f"Analyzing speakers...")
        phrases = self.speaker_classifier(audio_path)
        phrases.sort(key=lambda x: x["start"])
        return phrases

    def process_video(self, video_path: str, phrases: list) -> str:
        """Обрабатывает видео, добавляя аннотации и сохраняет результат."""
        logger.info(f"Drawing subtitles...")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        temp_video_path = "temp_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))

        current_frame = 0

        logging.info("Drawing new frames...")

        total_cap_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tqdm_bar = tqdm(total=total_cap_frames, desc="Processing video")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = current_frame / fps

            active_phrases = []
            i = 0
            while i < len(phrases):
                phrase = phrases[i]
                if phrase["start"] <= current_time <= phrase["end"]:
                    active_phrases.append(phrase)
                elif current_time > phrase["end"]:
                    del phrases[i]
                    i -= 1
                i += 1

            faces = self.person_detector(frame)
            names = faces["names"]
            if len(names) == 1 and len(active_phrases) == 1:
                name = names[0]
                speaker = list(active_phrases)[0]["speaker"]
                if name != PersonDetector.UNKNOWN_NAME:
                    if speaker in self.persons[name].voices:
                        self.persons[name].voices[speaker] += 1
                    else:
                        self.persons[name].voices[speaker] = 1
                # else:
                #     face = faces["locations"][0]
                #     top, right, bottom, left = face
                #     face_image = frame[top:bottom, left:right]
                #     cv2.imshow("Face", face_image)
                #     cv2.waitKey(0)
                #     self.persons[speaker] = Person(speaker, "", "")

            for i in range(len(active_phrases)):
                phrase = active_phrases[i]
                speaker = phrase["speaker"]
                best_match_voice_person = max(
                    self.persons.values(), key=lambda x: x.voices[speaker] if speaker in x.voices else 0
                )
                if speaker in best_match_voice_person.voices:
                    phrase["name"] = best_match_voice_person.name
                else:
                    phrase["name"] = speaker

            viz_text = "\n".join([f"{phrase['name']}: {phrase['text']}" for phrase in active_phrases])
            self.add_annotation_to_frame(frame, viz_text, GREEN_COLOR, FONT)

            for i in range(len(faces["names"])):
                frame = viz.draw_person_name(frame, names[i], faces["locations"][i])
                # frame = viz.draw_landmarks(frame, faces["landmarks"][i])

            out_video.write(frame)
            current_frame += 1
            tqdm_bar.update(1)

        cap.release()
        out_video.release()
        cv2.destroyAllWindows()

        return temp_video_path

    @staticmethod
    def add_annotation_to_frame(frame, speaker: str, color: tuple, font) -> None:
        """Добавляет аннотацию к кадру с использованием PIL для отображения текста."""
        # Конвертируем изображение из формата OpenCV в формат PIL
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Рисуем прямоугольник для фона текста
        draw.rectangle([(0, frame.shape[0] - 40), (frame.shape[1], frame.shape[0])], fill=(0, 0, 0))

        # Рисуем текст на изображении
        draw.text((10, 10), speaker, font=font, fill=(255, 255, 255, 0))

        # Конвертируем изображение обратно в формат OpenCV
        frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def merge_audio_and_video(video_path: str, audio_path: str, save_path: str) -> None:
        """Объединяет аудио и видео в один файл."""
        logger.info(f"Merging audio and video...")

        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(save_path, codec="libx264", audio_codec="aac")

    def __call__(self, video_path: str, save_path: str) -> None:
        temp_audio_path = "temp.wav"
        self.convert_video_to_audio(video_path, temp_audio_path)
        phrases = self.analyze_speakers(temp_audio_path)
        temp_video_path = self.process_video(video_path, phrases)
        self.merge_audio_and_video(temp_video_path, temp_audio_path, save_path)


def process_test():
    video_path = "../../data/video/test_1_min.mp4"
    analyzer = VideoAnalyzer()
    analyzer(video_path, save_path="../../data/predicts/test_1_min_output.mp4")


def process_our_video():
    video_path = "/home/werserk/ours_test.mp4"
    analyzer = VideoAnalyzer()
    analyzer(video_path, save_path="../../data/predicts/ours.mp4")


def process_final():
    video_path = "../../data/video/final.mp4"
    analyzer = VideoAnalyzer()
    analyzer(video_path, save_path="../../data/predicts/final.mp4")


def process_part_final():
    video_path = "../../data/video/final_360-380.mp4"
    analyzer = VideoAnalyzer()
    analyzer(video_path, save_path="../../data/predicts/final_360-380.mp4")


if __name__ == "__main__":
    process_part_final()
