import os

from app.utils.cut import trim_video

DATA_DIR = "../data"
trim_video(os.path.join(DATA_DIR, "video/test.mp4"), os.path.join(DATA_DIR, "video/test_001.mp4"), 360, 370)
