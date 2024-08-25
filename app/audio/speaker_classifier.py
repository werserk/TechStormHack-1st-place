import logging
from typing import List, Dict, Union

from pyannote.audio import Pipeline
from tqdm import tqdm

from app.audio.text_transcriber import TextTranscriber


class SpeakerClassifier:
    def __init__(self) -> None:
        logging.info("Initializing SpeakerClassifier...")
        self.model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="YOUR_HF_TOKEN_HERE",
        )
        self.transcriber = TextTranscriber()

    def __call__(self, audio_data: str) -> List[Dict[str, Union[str, float]]]:
        diarization_data = []
        diarization = self.model(audio_data)
        for segment, _, speaker in tqdm(
            diarization.itertracks(yield_label=True), total=len(diarization), desc="Speech transcribition"
        ):
            start_time = segment.start
            end_time = segment.end
            text = self.transcriber(audio_data, start_time, end_time)
            diarization_data.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "speaker": speaker,
                    "text": text,
                }
            )

        return diarization_data
