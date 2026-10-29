"""
newsbot/llm/ — LLM-фильтр новостей (DeepSeek, OpenAI-compatible API).
"""
from .filter import LLMFilter, FilterResult, route
from .client import (
    DeepSeekClient, LLMError, LLMTruncationError, BalanceInfo,
    LLMFatalError, LLMContextLengthError,
)

__all__ = [
    "LLMFilter", "FilterResult", "route",
    "DeepSeekClient", "LLMError", "LLMTruncationError",
    "LLMFatalError", "LLMContextLengthError",
    "BalanceInfo",
]