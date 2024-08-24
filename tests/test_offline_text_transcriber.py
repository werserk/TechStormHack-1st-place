from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline

# Инициализируем модель для диаризации
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN_HERE",
)

print("Connecting to Hugging Face...")
# Инициализируем модель для распознавания речи
asr = hf_pipeline("automatic-speech-recognition", use_auth_token="YOUR_HF_TOKEN_HERE")

# Укажите путь к вашему аудиофайлу
audio_file = "../data/audio/test_2_min.mp3"

print("Processing audio...")
# Применяем диаризацию спикеров
diarization = pipeline(audio_file)

# Обрабатываем и выводим текст для каждой реплики
for turn, _, speaker in diarization.itertracks(yield_label=True):
    # Вырезаем кусок аудио для конкретного спикера
    speaker_audio = pipeline.crop(audio_file, turn)

    # Преобразуем аудио в текст
    text = asr(speaker_audio)["text"]

    print(f"Speaker {speaker}: {text}")
