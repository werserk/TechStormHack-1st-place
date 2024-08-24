from typing import List, Dict, Union

from pyannote.audio import Pipeline


class SpeakerClassifier:
    def __init__(self) -> None:
        self.model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="YOUR_HF_TOKEN_HERE",
        )
        print("Speaker classifier initialized.")

    def __call__(self, audio_data: str) -> List[Dict[str, Union[str, float]]]:
        diarization_data = []
        diarization = self.model(audio_data)
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start_time = segment.start
            end_time = segment.end
            diarization_data.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "speaker": speaker,
                }
            )

        return diarization_data
