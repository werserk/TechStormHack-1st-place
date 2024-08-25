import os

from app.utils.cut import trim_video

DATA_DIR = "../data"
trim_video(os.path.join(DATA_DIR, "video/final.mp4"), os.path.join(DATA_DIR, "video/final_200-210.mp4"), 200, 210)
