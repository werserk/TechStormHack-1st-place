import os

from app.utils.cut import trim_video

DATA_DIR = "../data"
trim_video(os.path.join(DATA_DIR, "video/final.mp4"), os.path.join(DATA_DIR, "video/final_360-380.mp4"), 360, 380)
