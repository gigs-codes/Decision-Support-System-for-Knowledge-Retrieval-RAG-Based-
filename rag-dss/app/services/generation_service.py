"""
Generation service — multi-provider LLM wrapper.

Supports OpenAI (GPT-4o-mini default), Anthropic (Claude), and a local
Ollama-compatible endpoint. Configured via LLM_PROVIDER env var.
"""

import time
from typing import AsyncIterator, List, Optional

import httpx

from app.core.config import settings
from app.core.logging import get_logger
from app.models.query import ConversationTurn, RetrievedChunk

logger = get_logger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert decision-support assistant with access to a curated knowledge base.
Your role is to provide accurate, well-reasoned answers grounded strictly in the retrieved context.

Guidelines:
- Answer ONLY from the provided context. If the context is insufficient, say so clearly.
- Cite sources by referencing [Source N] in your response.
- Be concise but thorough. Use bullet points or numbered lists where appropriate.
- End with 2-3 follow-up questions the user might find helpful.
- If the question involves a decision, present pros/cons or a structured analysis.
"""

RAG_PROMPT_TEMPLATE = """RETRIEVED CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

Instructions: Answer based solely on the retrieved context above. 
Cite sources as [Source 1], [Source 2], etc. 
After your answer, suggest 2-3 follow-up questions under the heading "Follow-up Questions:"."""


def _build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}] (doc: {chunk.document_title}, relevance: {chunk.score:.2f})\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def _build_history(turns: List[ConversationTurn], max_turns: int = 4) -> str:
    recent = turns[-max_turns * 2:]
    return "\n".join(f"{t.role.upper()}: {t.content}" for t in recent) or "None"


def _extract_follow_ups(answer: str) -> tuple[str, List[str]]:
    """Split answer text from the 'Follow-up Questions:' section."""
    marker = "follow-up questions:"
    lower = answer.lower()
    idx = lower.find(marker)
    if idx == -1:
        return answer.strip(), []
    body = answer[:idx].strip()
    tail = answer[idx + len(marker):].strip()
    questions = [
        line.lstrip("0123456789.-) ").strip()
        for line in tail.splitlines()
        if line.strip()
    ]
    return body, questions[:3]


class GenerationService:
    """Unified LLM interface."""

    async def generate(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        history: Optional[List[ConversationTurn]] = None,
    ) -> tuple[str, List[str], float]:
        """
        Returns (answer_text, follow_up_questions, latency_ms).
        """
        context = _build_context(chunks)
        hist_text = _build_history(history or [])
        user_prompt = RAG_PROMPT_TEMPLATE.format(
            context=context, history=hist_text, question=question
        )

        start = time.perf_counter()
        provider = settings.LLM_PROVIDER

        if provider == "openai":
            raw = await self._call_openai(user_prompt)
        elif provider == "anthropic":
            raw = await self._call_anthropic(user_prompt)
        else:
            raw = await self._call_local(user_prompt)

        latency = (time.perf_counter() - start) * 1000
        answer, follow_ups = _extract_follow_ups(raw)
        return answer, follow_ups, latency

    # ── OpenAI ────────────────────────────────────────────────────────────────

    async def _call_openai(self, user_prompt: str) -> str:
        import openai

        client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content or ""

    # ── Anthropic ─────────────────────────────────────────────────────────────

    async def _call_anthropic(self, user_prompt: str) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        message = await client.messages.create(
            model=settings.ANTHROPIC_MODEL,
            max_tokens=settings.LLM_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    # ── Local (Ollama) ────────────────────────────────────────────────────────

    async def _call_local(self, user_prompt: str) -> str:
        payload = {
            "model": settings.LOCAL_LLM_MODEL,
            "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
            "stream": False,
            "options": {
                "temperature": settings.LLM_TEMPERATURE,
                "num_predict": settings.LLM_MAX_TOKENS,
            },
        }
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
            resp = await client.post(settings.LOCAL_LLM_URL, json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")
