import logging
from typing import List, Dict

from app.llm.yandexgpt import YandexGPTSession
from app.production.constants import BASIC_ANALYTICS_PROMPT, YandexGPTConfig
from app.video.person import Person


class MetricsCalculator:
    def __init__(self, persons: List[Person]) -> None:
        self.gpt = YandexGPTSession(api_key=YandexGPTConfig.api_key, system_prompt=BASIC_ANALYTICS_PROMPT)
        self.metrics = {str(person): {
            "initiative": 0,
            "productivity": 0,
            "successfulness": 0,
        } for person in persons}

    def __call__(self, messages: List[Dict[str, str]]) -> None:
        for message in messages:
            if message["name"] in self.metrics:
                self.metrics[message["name"]]["initiative"] += 1
        print("Asking GPT...")
        result = self.gpt.send_message(str(messages))
        print(result)
