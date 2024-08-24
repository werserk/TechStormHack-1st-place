import ffmpeg


def trim_video(input_file: str, output_file: str, start_time: float, end_time: float) -> None:
    """
    Обрезает видео по времени.

    :param input_file: Путь к исходному видеофайлу.
    :param output_file: Путь для сохранения обрезанного видео.
    :param start_time: Время начала обрезки в секундах.
    :param end_time: Время окончания обрезки в секундах.
    """
    input_video = ffmpeg.input(input_file)
    trimmed_video = input_video.trim(start=start_time, end=end_time).setpts("PTS-STARTPTS")
    trimmed_audio = input_video.filter_("atrim", start=start_time, end=end_time).filter_("asetpts", "PTS-STARTPTS")
    output = ffmpeg.output(trimmed_video, trimmed_audio, output_file)
    ffmpeg.run(output)


def trim_audio(input_file: str, output_file: str, start_time: float, end_time: float) -> None:
    """
    Извлекает и обрезает аудио по времени.

    :param input_file: Путь к исходному видеофайлу.
    :param output_file: Путь для сохранения обрезанного аудиофайла.
    :param start_time: Время начала обрезки в секундах.
    :param end_time: Время окончания обрезки в секундах.
    """
    input_video = ffmpeg.input(input_file)
    trimmed_audio = input_video.filter_("atrim", start=start_time, end=end_time).filter_("asetpts", "PTS-STARTPTS")
    output = ffmpeg.output(trimmed_audio, output_file)
    ffmpeg.run(output)
