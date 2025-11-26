# llm_client_openai.py

from __future__ import annotations
from typing import Any
import os
from openai import OpenAI


class OpenAILLMClient:
    """
    Simple OpenAI LLM client wrapper for the semantic intent parser.

    Usage:
        client = OpenAILLMClient(
            model_name="gpt-4.1-mini",
            temperature=0.0,
        )
        text = client.invoke(prompt)
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        # Use OPENAI_API_KEY from the environment
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def invoke(self, prompt: str) -> str:
        """
        Call the OpenAI Chat Completions API and return the assistant text.
        The prompt is already the combined "system+user" built by semantic_intent.py.
        """
        response = self._client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful model that follows the user instructions exactly.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        # Extract the assistant text
        content = response.choices[0].message.content
        return content if content is not None else ""
