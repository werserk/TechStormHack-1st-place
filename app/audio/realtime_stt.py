from RealtimeSTT import AudioToTextRecorder


class TextTranscriberOnline:
    def __init__(self):
        recorder_config = {
            "spinner": False,
            "model": "base",
            "silero_sensitivity": 0.4,
            "webrtc_sensitivity": 2,
            "post_speech_silence_duration": 0.4,
            "min_length_of_recording": 0,
            "min_gap_between_recordings": 0,
            "enable_realtime_transcription": True,
            "realtime_processing_pause": 0.2,
            "realtime_model_type": "base",
            "on_realtime_transcription_update": self.process_detected_text,
            "silero_deactivity_detection": True,
        }

        self.model = AudioToTextRecorder(**recorder_config)
        self.last_message = ""
        self.sentences = []

    def process_detected_text(self, text: str) -> None:
        new_text = "\n".join(self.sentences).strip() + "\n" + text if len(self.sentences) > 0 else text
        if new_text != self.last_message:
            displayed_text = new_text
            print("\n" * 20)
            print(f"Language: {self.model.detected_language} (realtime: {self.model.detected_realtime_language})")
            print(displayed_text, end="", flush=True)

    def process_text(self, text: str) -> None:
        self.sentences.append(text)
        self.process_detected_text("")

    def listen(self) -> None:
        print("\n" * 20)
        print("Listening...")
        while True:
            self.model.text(self.process_text)
