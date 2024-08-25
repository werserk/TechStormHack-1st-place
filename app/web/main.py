import os
import tempfile
import time

import numpy as np
import pandas as pd
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
                metrics = video_analyzer(video_path, output_path)

            with st.expander("Результат обработки"):
                column_names = ["Имя", "Конструктивность", "Инициативность", "ИПК"]
                list_mertics = [
                    [name, metrics[name]["constructive"], metrics[name]["initiative"], metrics[name]["IPC"]]
                    for name in metrics.keys()
                ]
                for i in range(len(list_mertics)):
                    for j in range(len(list_mertics[i])):
                        if np.isnan(list_mertics[i][j]) or list_mertics[i][j] == 0 or np.isinf(list_mertics[i][j]):
                            list_mertics[i][j] = "-"
                st.dataframe(pd.DataFrame(list_mertics, columns=column_names))

            # download
            st.download_button("Скачать видео", output_path, file_name=f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4")

            # Проигрывание обработанного видео
            if os.path.exists(output_path):
                st.video(output_path)
            else:
                st.warning("Обработанное видео не найдено!")
