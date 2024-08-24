import logging
import os
from typing import Optional

import cv2
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment

from app.audio.speaker_classifier import SpeakerClassifier
from app.audio.text_transcriber import TextTranscriberOffline
from app.video.detector import PersonDetector

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


class VideoAnalyzer:
    def __init__(self) -> None:
        self.transcriber = TextTranscriberOffline()
        self.speaker_classifier = SpeakerClassifier()
        self.person_detector = PersonDetector()

    def convert_video_to_audio(self, video_path: str, temp_audio_path: str) -> None:
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

    def process_video(self, video_path: str, phrases: list, save_path: Optional[str] = None) -> str:
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

            if len(phrases) > 0:
                speaker = phrases[0]["speaker"]
                color = self.get_speaker_color(speaker)
                self.add_annotation_to_frame(frame, speaker, color, font)
                if current_time >= phrases[0]["end"]:
                    phrases.pop(0)

            out_video.write(frame)
            current_frame += 1

        cap.release()
        out_video.release()
        cv2.destroyAllWindows()

        return temp_video_path

    def get_speaker_color(self, speaker: str) -> tuple:
        """Возвращает цвет аннотации в зависимости от спикера."""
        return 0, 0, 255

    def add_annotation_to_frame(self, frame, speaker: str, color: tuple, font) -> None:
        """Добавляет аннотацию к кадру."""
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), color, -1)
        cv2.putText(frame, speaker, (10, 30), font, 1, (255, 255, 255), 2)

    def merge_audio_and_video(self, video_path: str, audio_path: str, save_path: str) -> None:
        """Объединяет аудио и видео в один файл."""
        logger.info(f"Merging audio and video...")

        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(save_path, codec='libx264', audio_codec='aac')

    def __call__(self, video_path: str, save_path: Optional[str] = None) -> None:
        temp_audio_path = "temp.wav"
        self.convert_video_to_audio(video_path, temp_audio_path)
        phrases = self.analyze_speakers(temp_audio_path)
        temp_video_path = self.process_video(video_path, phrases)
        if save_path:
            self.merge_audio_and_video(temp_video_path, temp_audio_path, save_path)
        else:
            logger.error("Save path is required for merging video and audio.")


if __name__ == "__main__":
    video_path = "../../data/video/test_1_min.mp4"
    analyzer = VideoAnalyzer()
    analyzer(video_path, save_path="../../data/video/test_1_min_output.mp4")
