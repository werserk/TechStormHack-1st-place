FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


RUN apt-get install cmake -y
RUN apt-get install build-essential -y
RUN apt-get install libgtk-3-dev libboost-all-dev libportaudio2 portaudio19-dev -y

RUN pip install dlib
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install

EXPOSE 8501