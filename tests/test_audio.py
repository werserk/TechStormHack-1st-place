import numpy as np
import pyaudio
import whisper

# Step 2: Load the Whisper model
model = whisper.load_model("base")  # You can use other model sizes like "small", "medium", "large"

# Step 3: Set up PyAudio for real-time audio capture
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper prefers 16kHz
CHUNK = 1024  # Number of frames per buffer

audio = pyaudio.PyAudio()

# Step 4: Start the audio stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

print("Listening...")

try:
    while True:
        # Step 5: Read audio data from the stream
        audio_data = stream.read(CHUNK, exception_on_overflow=False)

        # Convert the byte data to numpy array
        np_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Step 6: Transcribe the audio data using Whisper
        result = model.transcribe(np_data, fp16=False)

        # Step 7: Print the transcribed text
        print("Transcription:", result['text'])

except KeyboardInterrupt:
    print("Terminating...")
finally:
    # Step 8: Clean up
    stream.stop_stream()
    stream.close()
    audio.terminate()
