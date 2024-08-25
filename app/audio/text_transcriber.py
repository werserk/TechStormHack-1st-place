import logging
import os
import tempfile

from faster_whisper import WhisperModel
from pydub import AudioSegment


class TextTranscriber:
    def __init__(self, model_name: str = "base"):
        logging.info("Initializing TextTranscriber...")
        self.model = WhisperModel(model_name, compute_type="float16")

    def _extract_audio_segment(self, audio_path: str, start_time: float, end_time: float) -> str:
        """
        Извлекает сегмент аудиофайла.
        """
        # Загрузка аудиофайла
        audio = AudioSegment.from_file(audio_path)

        # Конвертирование времени в миллисекунды
        start_ms = start_time * 1000
        end_ms = end_time * 1000

        # Обрезка аудиофайла
        segment = audio[start_ms:end_ms]

        # Сохранение временного файла
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        segment.export(temp_file.name, format="wav")

        return temp_file.name

    def __call__(self, audio_path: str, start_time: float, end_time: float) -> str:
        """
        Транскрибирует часть аудиофайла в заданном диапазоне времени.
        """
        # Извлечение сегмента аудиофайла
        segment_path = self._extract_audio_segment(audio_path, start_time, end_time)

        try:
            # Выполнение транскрипции
            segments, _ = self.model.transcribe(segment_path, language="ru", beam_size=3)

            # Объединение всех сегментов в один текст
            full_text = " ".join(segment.text for segment in segments)
        finally:
            # Удаление временного файла
            os.remove(segment_path)

        return full_text
