"""
LLM-фильтр новостей.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..collector import NewsItem
from .client import DeepSeekClient, LLMError, BalanceInfo
from .prompts import SYSTEM_PROMPT, build_feedback_block

logger = logging.getLogger("newsbot.llm.filter")


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------


@dataclass
class FilterResult:
    """
    Результат оценки одной новости LLM-фильтром.

    Attributes
    ----------
    ok : bool
        True — фильтр отработал валидно (score валиден, можно принимать решение).
        False — ошибка LLM или невалидный ответ → fallback: слать непрофильтрованным.
    score : Optional[int]
        Оценка 0–10. None если ok=False.
    summary : str
        Выжимка фактов. Пустая если ok=False.
    reason : str
        Краткое обоснование. Пустая если ok=False.
    """

    ok: bool
    score: Optional[int] = None
    summary: str = ""
    reason: str = ""


# ----------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------


def route(score: int, push_threshold: int, digest_threshold: int) -> str:
    """
    Решает, куда отправить новость по числовой оценке.

    Parameters
    ----------
    score : int
        Оценка LLM, 0–10.
    push_threshold : int
        Порог немедленного пуша (score >= push_threshold → push).
    digest_threshold : int
        Порог дайджеста (push_threshold > score >= digest_threshold → digest).
        score < digest_threshold → skip.

    Returns
    -------
    str
        "push" | "digest" | "skip"
    """
    if score >= push_threshold:
        return "push"
    elif score >= digest_threshold:
        return "digest"
    else:
        return "skip"


# ----------------------------------------------------------------------
# LLMFilter
# ----------------------------------------------------------------------


class LLMFilter:
    """
    Оценивает NewsItem через LLM (DeepSeek) и возвращает FilterResult.

    Parameters
    ----------
    config_llm : dict
        Секция config["llm"] — provider, model, api_key, base_url,
        temperature, timeout_sec, max_retries, push_threshold, digest_threshold,
        system_prompt.
    client : DeepSeekClient, optional
        Клиент. Если None — создаётся из config_llm.
    """

    def __init__(
        self,
        config_llm: Dict[str, Any],
        client: Optional[DeepSeekClient] = None,
    ):
        self.enabled = config_llm.get("enabled", True)
        self.model = config_llm.get("model", "")
        self.push_threshold = int(config_llm.get("push_threshold", 7))
        self.digest_threshold = int(config_llm.get("digest_threshold", 5))
        self.system_prompt = config_llm.get("system_prompt") or SYSTEM_PROMPT

        if client is not None:
            self._client = client
        else:
            self._client = DeepSeekClient(
                api_key=config_llm.get("api_key", ""),
                base_url=config_llm.get("base_url", "https://api.deepseek.com"),
                model=config_llm.get("model", ""),
                timeout=float(config_llm.get("timeout_sec", 60)),
                max_retries=int(config_llm.get("max_retries", 3)),
                temperature=float(config_llm.get("temperature", 0.0)),
            )

    def evaluate(
        self,
        item: NewsItem,
        feedback_context: Optional[tuple] = None,
    ) -> FilterResult:
        """
        Оценивает одну новость.

        Parameters
        ----------
        item : NewsItem
            Новость для оценки.
        feedback_context : tuple, optional
            Кортеж (dislikes, likes), где каждый элемент — список dict
            из get_recent_feedback(). Если передан — блок обратной связи
            подставляется в user-сообщение перед новостью.
            При ошибке получения фидбэка — best-effort, вызывающий код
            уже отловил исключение и передал пустые списки.

        При ошибке LLM — возвращает FilterResult(ok=False), что означает
        «fallback: слать непрофильтрованным». Исключение не бросается.
        """
        if not self.enabled:
            return FilterResult(ok=True, score=10, summary=item.summary, reason="llm disabled")

        user_message = self._build_user_message(item, feedback_context=feedback_context)

        try:
            raw = self._client.chat_json(
                system=self.system_prompt,
                user=user_message,
            )
            return self._parse_response(raw)

        except LLMError as e:
            logger.warning(
                "LLM-ошибка при оценке '%s': %s — fallback: шлём непрофильтрованным",
                item.title[:60],
                e,
            )
            return FilterResult(ok=False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        item: NewsItem,
        feedback_context: Optional[tuple] = None,
    ) -> str:
        """
        Формирует user-сообщение для LLM из полей NewsItem.

        Parameters
        ----------
        item : NewsItem
        feedback_context : tuple, optional
            (dislikes, likes) — списки dict из get_recent_feedback().
            Если не None и не пуст — перед новостью вставляется блок
            обратной связи, ЯВНО промаркированный как данные.
        """
        parts: List[str] = []

        # Блок обратной связи — перед новостью (few-shot, Фаза 2c)
        if feedback_context is not None:
            dislikes, likes = feedback_context
            feedback_block = build_feedback_block(
                dislikes=dislikes,
                likes=likes,
                max_chars_per_example=200,
            )
            if feedback_block:
                parts.append(feedback_block)
                parts.append("")  # пустая строка как разделитель

        parts.append(f"# Заголовок\n{item.title}")

        if item.summary:
            parts.append(f"# Содержание\n{item.summary}")

        if item.source:
            parts.append(f"# Источник: {item.source}")

        return "\n\n".join(parts)

    # ── balance ─────────────────────────────────────────────────────────────

    def get_balance(self) -> BalanceInfo:
        """Делегирует клиенту. Возвращает BalanceInfo; исключений не бросает."""
        return self._client.get_balance()

    @property
    def client(self) -> DeepSeekClient:
        """Доступ к HTTP-клиенту для pipeline / close."""
        return self._client

    # ── internal ────────────────────────────────────────────────────────────

    def _parse_response(self, raw: Dict[str, Any]) -> FilterResult:
        """
        Валидирует JSON-ответ LLM и возвращает FilterResult.

        T2-9 (трейд-офф):
          Принимаем: int; float если целочисленный (7.0); str если парсится в целое ("7").
          Отклоняем: bool (подкласс int — проверяем ДО int); NaN/inf/значения вне 0..10;
          нечисловые строки.
          Комментарий: модель иногда возвращает 7.0 (float) или "7" (str) —
          это рабочие значения, не мусор. Принимаем их вместо fallback'а.
        """
        raw_score = raw.get("score")
        score: Optional[int] = None

        # ── T2-9: порядок ВАЖЕН — bool перед int (bool подкласс в Python) ──
        if isinstance(raw_score, bool):
            # bool отклоняем (True→1, False→0 — это не число, а флаг)
            pass
        elif isinstance(raw_score, int) and not isinstance(raw_score, bool):
            # int (но не bool, т.к. bool уже отсечён)
            score = raw_score
        elif isinstance(raw_score, float):
            # float: принимаем если целочисленный и в диапазоне 0..10
            import math
            if (math.isfinite(raw_score)
                    and float(int(raw_score)) == raw_score
                    and 0 <= raw_score <= 10):
                score = int(raw_score)
            # NaN/inf/дробные — не принимаем
        elif isinstance(raw_score, str):
            s = raw_score.strip()
            try:
                # принимаем str если парсится в целое в диапазоне
                candidate = int(s)
                if 0 <= candidate <= 10:
                    score = candidate
            except (ValueError, TypeError):
                pass  # нечисловая строка → не принимаем

        if score is None or not (0 <= score <= 10):
            logger.warning(
                "LLM вернул score %r (не int 0..10) → fallback: шлём непрофильтрованным",
                raw_score,
            )
            return FilterResult(ok=False)

        summary = str(raw.get("summary", "")).strip()
        if not summary:
            logger.warning(
                "LLM вернул пустой summary → fallback: шлём непрофильтрованным"
            )
            return FilterResult(ok=False)

        reason = str(raw.get("reason", "")).strip()

        return FilterResult(
            ok=True,
            score=score,
            summary=summary,
            reason=reason,
        )