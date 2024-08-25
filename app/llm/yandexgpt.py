import requests


class YandexGPTSession:
    def __init__(
        self,
        api_key,
        system_prompt,
        model_uri="gpt://b1g73v4ajgb1ghai7uk6/yandexgpt-lite",
        temperature=0.6,
        max_tokens=2000,
    ):
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.headers = {"Content-Type": "application/json", "Authorization": f"Api-Key {api_key}"}
        self.model_uri = model_uri
        self.completion_options = {"stream": False, "temperature": temperature, "maxTokens": str(max_tokens)}
        # Инициализация с системным промптом
        self.messages = [{"role": "system", "text": system_prompt}]

    def send_message(self, user_message):
        """Отправляет сообщение пользователем и получает ответ от модели."""
        # Добавляем сообщение пользователя в список сообщений
        self.messages.append({"role": "user", "text": user_message})

        prompt = {"modelUri": self.model_uri, "completionOptions": self.completion_options, "messages": self.messages}

        response = requests.post(self.api_url, headers=self.headers, json=prompt)
        if response.status_code == 200:
            response_data = response.json()
            assistant_message = response_data["result"]["alternatives"][0]["message"]["text"]
            self.messages.append({"role": "assistant", "text": assistant_message})
            return assistant_message
        else:
            print("Error:", response.status_code, response.text)
            return None
