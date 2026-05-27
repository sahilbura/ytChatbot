import os
from typing import List

import requests


class GroqEmbeddings:
    """Minimal embeddings adapter for Groq API.

    Provides `embed_documents` and `embed_query` used by LangChain-style callers.
    Configure the key with the `GROQ_API_KEY` environment variable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        model: str = "text-embedding-3-small",
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        # Groq exposes OpenAI-compatible routes under /openai/v1.
        self.endpoint = endpoint or os.getenv(
            "GROQ_EMBED_ENDPOINT", "https://api.groq.com/openai/v1/embeddings"
        )
        self.model = model

        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment; cannot use GroqEmbeddings")

    def _request(self, inputs: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {"model": self.model, "input": inputs}

        resp = requests.post(self.endpoint, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        j = resp.json()

        # Expecting response like: { data: [ { embedding: [...] }, ... ] }
        items = j.get("data") or j.get("embeddings") or j
        embeddings = []
        for entry in items:
            if isinstance(entry, dict) and "embedding" in entry:
                embeddings.append(entry["embedding"])
            elif isinstance(entry, list):
                embeddings.append(entry)
            else:
                raise ValueError("Unexpected Groq embeddings response format")

        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._request(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._request([text])[0]
