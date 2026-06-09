"""Conversation memory helpers for adaptive follow-up retrieval."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)

REFERENCE_WORDS = {"it", "its", "that", "those", "them", "previous", "earlier", "before"}


def build_conversation_memory(chat_history: Iterable[Dict[str, Any]], window: int = 3):
    """Build a ConversationBufferMemory object from recent Streamlit chat history."""
    try:
        from langchain.memory import ConversationBufferMemory
    except Exception:  # pragma: no cover - LangChain version compatibility.
        return None

    memory = ConversationBufferMemory(return_messages=False, input_key="question", output_key="answer")
    for item in list(chat_history or [])[-window:]:
        try:
            memory.save_context({"question": item.get("question", "")}, {"answer": item.get("answer", "")})
        except Exception:
            logger.exception("Could not save one chat turn into ConversationBufferMemory.")
    return memory


def detect_conversational_references(query: str) -> bool:
    """Return True when the user is referring to prior conversation context."""
    tokens = set(re.findall(r"[a-zA-Z0-9']+", (query or "").lower()))
    return bool(tokens.intersection(REFERENCE_WORDS)) or "previous one" in (query or "").lower()


def get_recent_memory_context(chat_history: Iterable[Dict[str, Any]], window: int = 3) -> str:
    """Format the last few turns for memory-aware prompting."""
    try:
        turns = list(chat_history or [])[-window:]
        lines: List[str] = []
        for item in turns:
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            if question:
                lines.append(f"User: {question}")
            if answer:
                lines.append(f"Assistant: {answer[:900]}")
        return "\n".join(lines)
    except Exception:
        logger.exception("Memory formatting failed. Continuing without memory.")
        return ""


def get_previous_chunks(chat_history: Iterable[Dict[str, Any]], window: int = 3) -> List[Dict[str, Any]]:
    """Return previously retrieved source chunks, newest first."""
    chunks: List[Dict[str, Any]] = []
    try:
        for item in reversed(list(chat_history or [])[-window:]):
            chunks.extend(item.get("retrieved_chunks", []) or [])
    except Exception:
        logger.exception("Previous chunk lookup failed. Continuing without memory chunks.")
    return chunks


def get_memory_window(query: str, chat_history: Iterable[Dict[str, Any]], confidence: float = 0.8) -> int:
    """Use last 3 turns by default; expand to 5 for ambiguous low-confidence follow-ups."""
    if detect_conversational_references(query) and confidence < 0.55 and list(chat_history or []):
        return 5
    return 3
