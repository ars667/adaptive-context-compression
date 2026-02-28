"""Groq API client for LLM inference."""

from typing import Dict
from groq import Groq
from src.config import get_settings


class GroqClient:
    """Client for Groq API with llama-3.1-8b-instant."""

    def __init__(self):
        settings = get_settings()
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.LLM_MODEL

    def generate(self, context: str, query: str) -> Dict:
        """
        Generate answer using Groq API.

        Args:
            context: Retrieved and possibly compressed context
            query: User question

        Returns:
            Dict with answer and tokens_used
        """
        system_prompt = (
            "Ты помощник студента. Отвечай на вопросы строго опираясь на предоставленный "
            "контекст из учебника. Если информации нет в контексте — скажи об этом. "
            "Отвечай на том же языке, на котором задан вопрос."
        )

        # Build messages for chat completion
        if context:
            user_message = f"Контекст из учебника:\n\n{context}\n\nВопрос: {query}"
        else:
            user_message = f"Вопрос: {query}\n\n(Контекст не предоставлен)"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Low temperature for factual answers
                max_tokens=1024,
            )

            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0

            return {"answer": answer, "tokens_used": tokens_used}

        except Exception as e:
            print(f"❌ Groq API error: {e}")
            return {
                "answer": f"Ошибка при получении ответа: {str(e)}",
                "tokens_used": 0,
            }
