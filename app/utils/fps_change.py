import cv2


def change_video_fps(input_video_path: str, output_video_path: str, desired_fps=24):
    cap = cv2.VideoCapture(input_video_path)

    # Получаем исходные параметры видео
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Настраиваем выходное видео
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # или 'XVID' для .avi формата
    out = cv2.VideoWriter(output_video_path, fourcc, desired_fps, (frame_width, frame_height))

    # Пересчитываем шаг для пропуска фреймов
    frame_skip = int(original_fps / desired_fps)
    print(f"Original FPS: {original_fps:.2f}")
    print(f"Desired FPS: {desired_fps:.2f}")

    # Итерация по фреймам
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Записываем каждый n-й фрейм в зависимости от нового FPS
        if frame_count % frame_skip == 0:
            out.write(frame)

        frame_count += 1

    # Освобождаем ресурсы
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    change_video_fps("../../data/video/test_1_min.mp4", "../data/video/test_1_min_24_fps.mp4", 24)
