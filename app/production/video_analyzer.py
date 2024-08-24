import logging
import os

import cv2
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment

from app.audio.speaker_classifier import SpeakerClassifier
from app.audio.text_transcriber import TextTranscriberOffline
from app.video.detector import PersonDetector

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

RED_COLOR = (0, 0, 255)


class VideoAnalyzer:
    def __init__(self) -> None:
        self.transcriber = TextTranscriberOffline()
        self.speaker_classifier = SpeakerClassifier()
        self.person_detector = PersonDetector()

    @staticmethod
    def convert_video_to_audio(video_path: str, temp_audio_path: str) -> None:
        """Конвертирует видеофайл в аудиофайл."""
        logger.info(f"Converting video to audio...")
        audio = AudioSegment.from_file(video_path, format="mp4")
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
        font = cv2.FONT_HERSHEY_SIMPLEX

        current_frame = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = current_frame / fps

            active_speakers = []
            i = 0
            while i < len(phrases):
                phrase = phrases[i]
                if phrase["start"] <= current_time <= phrase["end"]:
                    active_speakers.append(phrase["speaker"])
                elif current_time > phrase["end"]:
                    del phrases[i]
                    i -= 1
                i += 1

            self.add_annotation_to_frame(frame, ", ".join(active_speakers), RED_COLOR, font)

            out_video.write(frame)
            current_frame += 1

        cap.release()
        out_video.release()
        cv2.destroyAllWindows()

        return temp_video_path

    @staticmethod
    def add_annotation_to_frame(frame, speaker: str, color: tuple, font) -> None:
        """Добавляет аннотацию к кадру."""
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), color, -1)
        cv2.putText(frame, speaker, (10, 30), font, 1, (255, 255, 255), 2)

    @staticmethod
    def merge_audio_and_video(video_path: str, audio_path: str, save_path: str) -> None:
        """Объединяет аудио и видео в один файл."""
        logger.info(f"Merging audio and video...")

        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(save_path, codec='libx264', audio_codec='aac')

    def __call__(self, video_path: str, save_path: str) -> None:
        temp_audio_path = "temp.wav"
        self.convert_video_to_audio(video_path, temp_audio_path)
        phrases = self.analyze_speakers(temp_audio_path)
        temp_video_path = self.process_video(video_path, phrases)
        self.merge_audio_and_video(temp_video_path, temp_audio_path, save_path)


if __name__ == "__main__":
    video_path = "../../data/video/test_1_min.mp4"
    analyzer = VideoAnalyzer()
    analyzer(video_path, save_path="../../data/video/test_1_min_output.mp4")
