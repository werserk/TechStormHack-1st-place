import os
import tempfile

import streamlit as st

from app.production import VideoAnalyzer

video_analyzer = VideoAnalyzer()


def start_web():
    st.title("Видео обработка")

    # Загрузка видео
    uploaded_file = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Создаем временный файл для сохранения загруженного видео
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name
            print(f"Видео загружено: {video_path}")

        st.video(video_path)

        # Кнопка для обработки видео
        if st.button("Обработать видео"):
            # Временный файл для сохранения обработанного видео
            output_path = "output_" + os.path.basename(video_path) + ".mp4"

            # Вызов функции обработки видео
            with st.spinner("Обрабатываем видео..."):
                video_analyzer(video_path, output_path)

            # Проигрывание обработанного видео
            if os.path.exists(output_path):
                st.video(output_path)
            else:
                st.warning("Обработанное видео не найдено!")
