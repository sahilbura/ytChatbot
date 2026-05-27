import os
from typing import Any

import requests


class GroqChatResponse:
    def __init__(self, content: str):
        self.content = content


class GroqChat:
    """Minimal Groq chat adapter providing `invoke(prompt)`.

    Uses `GROQ_API_KEY` and optional `GROQ_CHAT_ENDPOINT` env var.
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        model: str = "llama-3.1-8b-instant",
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.endpoint = endpoint or os.getenv(
            "GROQ_CHAT_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions"
        )
        self.model = model

        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not set; cannot use GroqChat")

    def invoke(self, prompt: str) -> GroqChatResponse:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
        }

        resp = requests.post(self.endpoint, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        j = resp.json()

        # Expected shape: {choices: [{message: {content: "..."}}]}
        text = ""
        choices = j.get("choices") or []
        if choices and isinstance(choices, list):
            first = choices[0]
            msg = first.get("message") or first.get("text")
            if isinstance(msg, dict):
                text = msg.get("content", "")
            elif isinstance(msg, str):
                text = msg

        return GroqChatResponse(text)
