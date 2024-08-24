from app.utils.cut import trim_audio

audio_path = "../data/audio/test.mp3"
output_path = "../data/audio/test_2_min.mp3"
trim_audio(audio_path, output_path, 0, 120)
