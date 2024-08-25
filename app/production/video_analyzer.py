import logging
import os
from typing import Dict

import cv2
import numpy as np
from PIL import ImageDraw, Image
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from tqdm import tqdm
from transformers import pipeline

import app.video.viz as viz
from app.audio.speech_analyzer import SpeechAnalyzer
from app.production.constants import persons_part2, FONT
from app.video.detector import PersonDetector

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

DATA_DIR = "../data"
labels = ["constructive", "destructive"]


class VideoAnalyzer:
    def __init__(self) -> None:
        persons = persons_part2
        self.persons = {person.name: person for person in persons}
        self.speaker_classifier = SpeechAnalyzer()
        self.person_detector = PersonDetector(persons=persons)
        self.bert = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.messages = []

    @staticmethod
    def convert_video_to_audio(video_path: str, temp_audio_path: str) -> None:
        """Конвертирует видеофайл в аудиофайл."""
        logger.info(f"Converting video to audio...")
        try:
            audio = AudioSegment.from_file(video_path, format="mp4")
        except ValueError:
            audio = AudioSegment.from_file(video_path, format=video_path.split(".")[-1])
        audio.export(temp_audio_path, format="wav")

    def analyze_speakers(self, audio_path: str) -> list:
        """Анализирует аудио и возвращает фразы со временем и спикерами."""
        logger.info(f"Analyzing speakers...")
        phrases = self.speaker_classifier(audio_path)
        phrases.sort(key=lambda x: x["start"])
        return phrases

    def process_video(self, video_path: str, phrases: list) -> str:
        """Обрабатывает видео, добавляя аннотации и сохраняет результат."""
        logger.info(f"Drawing subtitles...")

        temp_video_path = "temp_video.mp4"

        cap, out_video, fps, frame_width, frame_height = self._initialize_video_processing(video_path, temp_video_path)

        self._process_frames(cap, out_video, fps, phrases)

        cap.release()
        out_video.release()
        cv2.destroyAllWindows()

        return temp_video_path

    def _initialize_video_processing(self, video_path: str, temp_video_path: str):
        """Инициализирует захват видео и видео писатель."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))

        return cap, out_video, fps, frame_width, frame_height

    def _process_frames(self, cap, out_video, fps, phrases):
        """Обрабатывает кадры и добавляет аннотации."""
        current_frame = 0
        total_cap_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tqdm_bar = tqdm(total=total_cap_frames, desc="Processing video")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = current_frame / fps

            active_phrases = self._get_active_phrases(phrases, current_time)
            for phrase in active_phrases:
                if phrase not in self.messages:
                    if phrase["name"] not in self.persons:
                        continue
                    self.messages.append(phrase)
                    constructive_value = self.bert(phrase["text"], candidate_labels=labels)["scores"][0]
                    self.persons[phrase["name"]].metrics["constructive"].append(constructive_value)
                    self.persons[phrase["name"]].metrics["count"] += 1

            self._update_persons_voices(frame, active_phrases)
            frame = Image.fromarray(frame)
            frame = self._annotate_frame(frame, active_phrases)
            frame = np.array(frame)
            frame = self._draw_faces(frame)

            out_video.write(frame)
            current_frame += 1
            tqdm_bar.update(1)

    def _get_active_phrases(self, phrases: list, current_time: float) -> list:
        """Возвращает активные фразы для текущего времени."""
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
        return active_phrases

    def _update_persons_voices(self, frame: np.ndarray, active_phrases: list) -> None:
        """Обновляет голоса для известных персон."""
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

        for i in range(len(active_phrases)):
            phrase = active_phrases[i]
            speaker = phrase["speaker"]
            best_match_voice_person = max(self.persons.values(), key=lambda x: x.voices.get(speaker, 0))
            phrase["name"] = best_match_voice_person.name if speaker in best_match_voice_person.voices else speaker

    def _annotate_frame(self, frame: Image, active_phrases: list) -> Image:
        """Добавляет аннотации к кадру."""
        viz_text = "\n".join([f"{phrase['name']}: {phrase['text']}" for phrase in active_phrases])
        return self.add_annotation_to_frame(frame, viz_text, FONT)

    def _draw_faces(self, frame: np.ndarray) -> np.ndarray:
        """Рисует имена людей на кадре."""
        faces = self.person_detector(frame)
        names = faces["names"]
        frame = Image.fromarray(frame)
        for i in range(len(faces["names"])):
            frame = viz.draw_person_name(frame, names[i], faces["locations"][i])
        return np.array(frame)

    @staticmethod
    def add_annotation_to_frame(frame: Image, speaker: str, font) -> Image:
        """Добавляет аннотацию к кадру с использованием PIL для отображения текста."""
        draw = ImageDraw.Draw(frame)
        draw.rectangle([(0, frame.size[1] - 40), (frame.size[0], frame.size[0])], fill=(0, 0, 0))
        draw.text((0, frame.size[1] - 40), speaker, font=font, fill=(255, 255, 255, 0))
        return frame

    @staticmethod
    def merge_audio_and_video(video_path: str, audio_path: str, save_path: str) -> None:
        """Объединяет аудио и видео в один файл."""
        logger.info(f"Merging audio and video...")

        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(save_path, codec="libx264", audio_codec="aac")

    def __call__(self, video_path: str, save_path: str) -> Dict[str, Dict[str, float]]:
        temp_audio_path = "temp.wav"
        self.convert_video_to_audio(video_path, temp_audio_path)
        phrases = self.analyze_speakers(temp_audio_path)
        temp_video_path = self.process_video(video_path, phrases)
        self.merge_audio_and_video(temp_video_path, temp_audio_path, save_path)
        return {
            str(person): {
                "constructive": float(np.mean(person.metrics["constructive"])),
                "count": person.metrics["count"],
                "IPC": np.log2(person.metrics["count"]) * float(np.mean(person.metrics["constructive"])),
            }
            for person in self.persons.values()
        }


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
