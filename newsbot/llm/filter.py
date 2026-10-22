"""
LLM-фильтр новостей.
"""
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from ..collector import NewsItem
from .client import DeepSeekClient, LLMError, LLMTruncationError, BalanceInfo
from .prompts import (
    SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT,
    build_feedback_block, build_batch_user_message, sanitize_item_field,
    DEDUP_SYSTEM_PROMPT, build_dedup_user_message,
    MAX_PROMPT_TITLE_CHARS, MAX_PROMPT_SUMMARY_CHARS, LANG_MARKER,
)

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
    urgent : bool
        Флаг срочности. True — событие мирового/национального масштаба
        (война, мобилизация, катастрофа с массовыми жертвами, ядерный инцидент).
        По умолчанию False (безопасный дефолт).
    cost_blocked : bool
        True — LLM не вызывался из-за превышения дневного лимита расходов.
        Это НЕ ошибка LLM: items должны остаться new/pending и быть
        обработаны в следующем цикле/на следующий день (лимит обнулится),
        а не уходить в deferred/fallback/failed.
    """

    ok: bool
    score: Optional[int] = None
    summary: str = ""
    reason: str = ""
    urgent: bool = False
    cost_blocked: bool = False


# ----------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------


def route(
    score: int,
    push_threshold: int,
    digest_threshold: int,
    urgent: bool = False,
    push_urgent_only: bool = False,
) -> str:
    """
    Решает, куда отправить новость по числовой оценке и флагу срочности.

    Parameters
    ----------
    score : int
        Оценка LLM, 0–10.
    push_threshold : int
        Порог немедленного пуша.
    digest_threshold : int
        Порог дайджеста (push_threshold > score >= digest_threshold → digest;
        score < digest_threshold → skip).
    urgent : bool
        Флаг срочности от LLM. При сомнении LLM → false (безопасный дефолт).
    push_urgent_only : bool
        Если True — пуш только при urgent=True AND score>=push_threshold.
        Если False — прежнее поведение: score>=push_threshold → push.

    Returns
    -------
    str
        "push" | "digest" | "skip"

    Логика при push_urgent_only=True:
        urgent AND score>=push_threshold → "push"
        urgent AND score<push_threshold  → "digest"   (urgent ниже push-порога → дайджест, не skip)
        not urgent AND score>=push_threshold → "digest"  (без urgent → digest, НЕ push)
        not urgent AND score>=digest_threshold → "digest"
        иначе → "skip"

    Логика при push_urgent_only=False (обратная совместимость):
        score>=push_threshold            → "push"
        score>=digest_threshold          → "digest"
        иначе                            → "skip"
    """
    if push_urgent_only:
        if urgent:
            # urgent НИКОГДА не skip — даже при очень низком score.
            # Нижняя граница для urgent = digest (не skip).
            if score >= push_threshold:
                return "push"
            else:
                return "digest"
        else:
            # не-urgent — стандартная логика по score
            if score >= push_threshold:
                return "digest"  # push_urgent_only=True: без urgent → digest, не push
            elif score >= digest_threshold:
                return "digest"
            else:
                return "skip"
    else:
        # прежнее поведение: score — единственный гейт
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
    storage : Storage, optional
        Хранилище. Если None — cost-guard отключён.
    """

    def __init__(
        self,
        config_llm: Dict[str, Any],
        client: Optional[DeepSeekClient] = None,
        storage=None,
    ):
        self.enabled = self._parse_bool(config_llm.get("enabled", True))
        self.model = config_llm.get("model", "")
        # Major #2-фикс: числовые поля парсятся защищённо — одно неверное
        # значение (например push_threshold: "high") не должно отключать
        # фильтр целиком. Невалидное значение → warning + дефолт по полю.
        self.push_threshold = self._parse_int_field(config_llm, "push_threshold", 7)
        self.digest_threshold = self._parse_int_field(config_llm, "digest_threshold", 5)

        # M1-фикс: при отсутствии ключа push_urgent_only — явный warning.
        # Дефолт True (urgent-only). Без warning'а legacy-конфиг без ключа
        # молча переключался бы в urgent-only (filter=True, route def=False).
        if "push_urgent_only" not in config_llm:
            logger.warning(
                "llm.push_urgent_only отсутствует в конфиге — "
                "используется дефолт True (urgent-only). "
                "Явно задайте push_urgent_only: true/false в config.yaml."
            )
        self.push_urgent_only = self._parse_bool(config_llm.get("push_urgent_only", True))
        self.system_prompt = config_llm.get("system_prompt") or SYSTEM_PROMPT

        # ── Task 3: cost guard config ────────────────────────────────
        self._pricing: Dict[str, Dict[str, float]] = {}
        pricing_conf = config_llm.get("pricing", {})
        if not isinstance(pricing_conf, dict):
            # Major 1-фикс: битая секция pricing не должна молча игнорироваться.
            logger.warning(
                "llm.pricing должен быть словарём, получено %s — cost-guard "
                "не сможет считать расходы",
                type(pricing_conf).__name__,
            )
        else:
            for model_name, prices in pricing_conf.items():
                if not isinstance(prices, dict):
                    logger.warning(
                        "llm.pricing[%r] должен быть словарём {input, output}, "
                        "получено %s — запись пропущена",
                        model_name, type(prices).__name__,
                    )
                    continue
                try:
                    self._pricing[model_name] = {
                        "input": float(prices.get("input", 0) or 0),
                        "output": float(prices.get("output", 0) or 0),
                    }
                except (TypeError, ValueError):
                    logger.warning(
                        "llm.pricing[%r] содержит нечисловые цены (%r/%r) — "
                        "запись пропущена",
                        model_name, prices.get("input"), prices.get("output"),
                    )
        self._daily_cost_limit = self._parse_float_field(
            config_llm, "daily_cost_limit", 0.0
        )
        # Контракт: отрицательный лимит = guard выключен.
        if self._daily_cost_limit < 0:
            logger.warning(
                "llm.daily_cost_limit=%r отрицательный — cost-guard выключен (0)",
                self._daily_cost_limit,
            )
            self._daily_cost_limit = 0.0

        # Storage для daily cost tracking
        self._storage = storage

        # Признак «последний _check_daily_cost выдал WARNING о лимите ВПЕРВЫЕ
        # за день» — используется _log_cost_blocked_pending, чтобы и
        # pending-напоминание писалось WARNING только один раз за день
        # (общий флаг llm_cost_limit_warning_done_YYYYMMDD на оба сообщения).
        self._cost_warning_fresh = False

        if client is not None:
            self._client = client
        else:
            prices = self._pricing.get(self.model, {})
            self._client = DeepSeekClient(
                api_key=config_llm.get("api_key", ""),
                base_url=config_llm.get("base_url", "https://api.deepseek.com"),
                model=self.model,
                timeout=self._parse_float_field(config_llm, "timeout_sec", 60.0),
                max_retries=self._parse_non_negative_int_field(
                    config_llm, "max_retries", 3
                ),
                temperature=self._parse_float_field(config_llm, "temperature", 0.0),
                max_tokens=self._parse_int_field(config_llm, "max_tokens", 4096),
                input_price=prices.get("input", 0),
                output_price=prices.get("output", 0),
            )

        # Major 1-фикс: cost-guard активен, но цены модели неизвестны →
        # дневной расход всегда 0.0 и лимит никогда не сработает.
        # Поведение не меняем — только явный warning ОДИН раз при старте.
        self._warn_if_cost_guard_blind()

    # ── защищённый парсинг числовых полей конфига (Major 2-фикс) ──────

    @staticmethod
    def _parse_int_field(
        config: Dict[str, Any], key: str, default: int
    ) -> int:
        """
        int-поле конфига: невалидное значение → warning + default.
        Никогда не бросает исключение (фильтр не должен отключаться
        из-за одной опечатки в конфиге).
        """
        raw = config.get(key, default)
        if isinstance(raw, bool):
            logger.warning(
                "llm.%s=%r — ожидалось целое число, используется дефолт %r",
                key, raw, default,
            )
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.warning(
                "llm.%s=%r — ожидалось целое число, используется дефолт %r",
                key, raw, default,
            )
            return default

    @classmethod
    def _parse_non_negative_int_field(
        cls, config: Dict[str, Any], key: str, default: int
    ) -> int:
        """Как _parse_int_field, но отрицательное значение → 0."""
        value = cls._parse_int_field(config, key, default)
        if value < 0:
            logger.warning(
                "llm.%s=%r отрицательное — используется 0", key, value
            )
            return 0
        return value

    @staticmethod
    def _parse_float_field(
        config: Dict[str, Any], key: str, default: float
    ) -> float:
        """
        float-поле конфига: невалидное значение → warning + default.
        Никогда не бросает исключение.
        """
        raw = config.get(key, default)
        if isinstance(raw, bool):
            logger.warning(
                "llm.%s=%r — ожидалось число, используется дефолт %r",
                key, raw, default,
            )
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "llm.%s=%r — ожидалось число, используется дефолт %r",
                key, raw, default,
            )
            return default

    def _warn_if_cost_guard_blind(self) -> None:
        """
        Major 1-фикс: если дневной лимит задан, а цены модели неизвестны
        (нет в pricing и в клиенте нули) — лимит никогда не сработает.
        Логируем warning ОДИН раз при старте; поведение guard не меняем.
        """
        if self._daily_cost_limit <= 0:
            return

        prices = self._pricing.get(self.model) or {}
        input_price = prices.get("input", 0) or 0
        output_price = prices.get("output", 0) or 0

        if not input_price and not output_price:
            stats = getattr(self._client, "usage_stats", None)
            input_price = float(getattr(stats, "_input_price", 0) or 0)
            output_price = float(getattr(stats, "_output_price", 0) or 0)

        if not input_price and not output_price:
            logger.warning(
                "cost-guard активен (daily_cost_limit=%.2f USD), но цены модели "
                "%r не заданы в llm.pricing — дневной лимит работать не будет "
                "(расход всегда 0.0)",
                self._daily_cost_limit, self.model,
            )

    @staticmethod
    def _parse_bool(value) -> bool:
        """
        Явный парсер bool-значений из конфига.
        - bool: как есть
        - str: "true","1","yes","on" (case-insensitive) → True; всё остальное → False
        - прочее: False
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return False

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

        usage_before = self._usage_totals()
        try:
            raw = self._client.chat_json(
                system=self.system_prompt,
                user=user_message,
            )
        except LLMError as e:
            # usage мог быть списан API (например finish_reason=length):
            # аккумулируем ДО выхода, чтобы дневной лимит учитывал эти токены.
            self._record_usage(usage_before)
            logger.warning(
                "LLM-ошибка при оценке '%s': %s — fallback: шлём непрофильтрованным",
                item.title[:60],
                e,
            )
            return FilterResult(ok=False)

        self._record_usage(usage_before)
        return self._parse_response(raw)

    def evaluate_batch(
        self,
        items: List["NewsItem"],
        feedback_context: Optional[tuple] = None,
    ) -> List[FilterResult]:
        """
        Оценивает батч новостей одним LLM-вызовом.

        Все items нумеруются индексами 1..N в user-сообщении; LLM
        возвращает JSON ``{"results": [{"id": 1, ...}, ...]}``.
        Маппинг по ``id`` → FilterResult для каждого входного item.
        Per-item fallback: невалидный/пропущенный/битый элемент → ``ok=False``
        для этого item (остальные обрабатываются normally).

        Полный сбой LLM (``LLMError``) → все items ``ok=False``.

        При ``llm.batch_size=1`` ведёт себя идентично ``evaluate()`` по одному item.

        Parameters
        ----------
        items : List[NewsItem]
            Список новостей для оценки (порядок сохраняется в выходном списке).
        feedback_context : tuple, optional
            (dislikes, likes) — списки dict из ``get_recent_feedback()``.
            Блок обратной связи вставляется один раз на батч.

        Returns
        -------
        List[FilterResult]
            ``len(results) == len(items)``; результаты в том же порядке,
            что и входные items. Каждый результат соответствует item с
            тем же индексом 1-based в ответе LLM.
        """
        if not items:
            return []

        if not self.enabled:
            return [
                FilterResult(
                    ok=True,
                    score=10,
                    summary=item.summary,
                    reason="llm disabled",
                )
                for item in items
            ]

        # Проверка cost guard перед вызовом (в т.ч. для batch_size=1)
        if not self._check_daily_cost():
            # Явный сигнал: LLM не вызывался из-за дневного лимита.
            # pipeline оставляет такие items в new/pending (не deferred/failed).
            self._log_cost_blocked_pending()
            return [
                FilterResult(ok=False, cost_blocked=True)
                for _ in items
            ]

        if len(items) == 1:
            result = self.evaluate(items[0], feedback_context=feedback_context)
            return [result]

        # Рекурсивный evaluate с length-split
        return self._evaluate_batch_recursive(items, feedback_context, depth=0)

    def _parse_batch_response(self, raw: Dict[str, Any], expected_count: int) -> List[FilterResult]:
        """
        Парсит батч-ответ LLM ``{"results": [...]}`` → List[FilterResult].

        Per-item валидация (как ``_parse_response``):
          - ``id`` — целое 1..N, если пропущен/битый → этот item ok=False.
          - ``score`` — int 0..10 (tolerant: целочисленный float, str-"int").
          - ``summary`` — непустой.
          - ``urgent`` — bool (tolerant: "true"/"false"/1/0).
          - Все поля валидны → ``FilterResult(ok=True, ...)``.

        При невалидном ``score``/``summary`` → ``ok=False`` для ЭТОГО item.
        Полный сбой (нет results, не dict) → все items ok=False.

        Parameters
        ----------
        raw : dict
            Распарсенный JSON от LLM.
        expected_count : int
            Количество входных items (длина батча).

        Returns
        -------
        List[FilterResult]
            Длина == expected_count. Результаты соответствуют входным items
            по индексу (id=1 → index 0, id=2 → index 1, ...).
        """
        # полный сбой структуры ответа
        if not isinstance(raw, dict):
            logger.warning(
                "LLM вернул не-объект в батч-ответе: %s → fallback для всех",
                type(raw).__name__,
            )
            return [FilterResult(ok=False) for _ in range(expected_count)]

        raw_results: Any = raw.get("results")
        if not isinstance(raw_results, list):
            logger.warning(
                "LLM вернул results не-списком: %s → fallback для всех",
                type(raw_results).__name__,
            )
            return [FilterResult(ok=False) for _ in range(expected_count)]

        # стартовый массив: все — ok=False (fallback по умолчанию)
        results: List[FilterResult] = [FilterResult(ok=False) for _ in range(expected_count)]

        # множество уже обработанных id (для детекции дубликатов)
        seen_ids: Set[int] = set()

        for item_raw in raw_results:
            if not isinstance(item_raw, dict):
                # битый элемент массива — пропускаем, ok=False уже стоит
                continue

            raw_id = item_raw.get("id")
            # парсим id: принимаем int; tolerant — целочисленный float/str
            parsed_id: Optional[int] = None
            if isinstance(raw_id, int) and not isinstance(raw_id, bool):
                if 1 <= raw_id <= expected_count:
                    parsed_id = raw_id
            elif isinstance(raw_id, float):
                if math.isfinite(raw_id) and float(int(raw_id)) == raw_id:
                    candidate = int(raw_id)
                    if 1 <= candidate <= expected_count:
                        parsed_id = candidate
            elif isinstance(raw_id, str):
                try:
                    candidate = int(raw_id.strip())
                    if 1 <= candidate <= expected_count:
                        parsed_id = candidate
                except (ValueError, TypeError):
                    pass

            if parsed_id is None:
                logger.warning(
                    "LLM вернул невалидный или out-of-range id %r → пропускаем",
                    raw_id,
                )
                continue  # ok=False уже стоит

            if parsed_id in seen_ids:
                logger.warning("LLM вернул дубликат id=%d → пропускаем", parsed_id)
                continue  # уже обработан (первый выигрывает), ok=False

            seen_ids.add(parsed_id)
            index = parsed_id - 1  # 1-based id → 0-based index

            # ── score (per-item валидация, tolerant) ─────────────────
            raw_score = item_raw.get("score")
            score: Optional[int] = None

            if isinstance(raw_score, bool):
                pass
            elif isinstance(raw_score, int) and not isinstance(raw_score, bool):
                score = raw_score
            elif isinstance(raw_score, float):
                if (math.isfinite(raw_score)
                        and float(int(raw_score)) == raw_score
                        and 0 <= raw_score <= 10):
                    score = int(raw_score)
            elif isinstance(raw_score, str):
                try:
                    candidate = int(raw_score.strip())
                    if 0 <= candidate <= 10:
                        score = candidate
                except (ValueError, TypeError):
                    pass

            if score is None or not (0 <= score <= 10):
                logger.warning(
                    "Батч item id=%d: score %r невалиден (не int 0..10) → ok=False",
                    parsed_id,
                    raw_score,
                )
                results[index] = FilterResult(ok=False)
                continue

            # ── summary (per-item валидация) ─────────────────────────
            # P4-фикс: `or ""` вместо дефолта в get() — JSON null → "None",
            # такой литерал проходил валидацию и уходил в дайджест.
            summary = str(item_raw.get("summary") or "").strip()
            if not summary:
                logger.warning(
                    "Батч item id=%d: пустой summary → ok=False",
                    parsed_id,
                )
                results[index] = FilterResult(ok=False)
                continue

            reason = str(item_raw.get("reason") or "").strip()

            # ── urgent (per-item валидация, tolerant) ─────────────────
            raw_urgent = item_raw.get("urgent")
            urgent: bool = False
            if isinstance(raw_urgent, bool):
                urgent = raw_urgent
            elif isinstance(raw_urgent, int) and not isinstance(raw_urgent, bool):
                urgent = bool(raw_urgent)
            elif isinstance(raw_urgent, str):
                s = raw_urgent.strip().lower()
                if s in ("true", "1", "yes", "on"):
                    urgent = True
                elif s in ("false", "0", "no", "off"):
                    urgent = False
                # остальное → False

            results[index] = FilterResult(
                ok=True,
                score=score,
                summary=summary,
                reason=reason,
                urgent=urgent,
            )

        return results

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

        title = sanitize_item_field(item.title)
        if len(title) > MAX_PROMPT_TITLE_CHARS:
            title = title[:MAX_PROMPT_TITLE_CHARS - 3] + "..."
        parts.append(f"# Title\n{title}")

        if item.summary:
            summary = sanitize_item_field(item.summary)
            if len(summary) > MAX_PROMPT_SUMMARY_CHARS:
                summary = summary[:MAX_PROMPT_SUMMARY_CHARS - 3] + "..."
            parts.append(f"# Content\n{summary}")

        if item.source:
            parts.append(f"# Source: {sanitize_item_field(item.source)}")

        # Language marker (injection-safe)
        lang = getattr(item, "lang", None)
        if lang:
            safe_lang = sanitize_item_field(str(lang))
            parts.append(LANG_MARKER.format(lang=safe_lang))

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

    # ── Task 3: daily cost guard ────────────────────────────────────────────

    def _check_daily_cost(self) -> bool:
        """
        Проверяет дневной лимит расходов.

        Returns:
            True — можно вызывать LLM; False — лимит превышен.
        При превышении — логирует WARNING ОДИН раз за день
        (через флаг в meta 'llm_cost_limit_warning_done_YYYYMMDD').
        """
        if self._daily_cost_limit <= 0 or not self._storage:
            return True

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        limit_key = f"llm_daily_cost_{today}"
        warning_flag_key = f"llm_cost_limit_warning_done_{today}"

        try:
            current_cost = float(self._storage.get_meta(limit_key) or "0")
        except (ValueError, TypeError):
            current_cost = 0.0

        # Решение о блокировке — ВСЕГДА по факту current_cost (не по флагу).
        # Флаг warning'а используется ТОЛЬКО для подавления повторного WARNING:
        # если оператор поднял daily_cost_limit в тот же UTC-день, блокировка
        # снимается сразу, без ожидания полуночи.
        if current_cost < self._daily_cost_limit:
            self._cost_warning_fresh = False
            return True

        if self._storage.get_meta(warning_flag_key) != "1":
            logger.warning(
                "LLM daily cost limit reached: $%.2f >= $%.2f — "
                "skipping LLM call (items will be retried later)",
                current_cost, self._daily_cost_limit,
            )
            # Ставим флаг что warning уже дан (повторно не спамим)
            self._storage.set_meta(warning_flag_key, "1")
            self._cost_warning_fresh = True
        else:
            # Повторное превышение в тот же день (в т.ч. после того как оператор
            # поднял лимит, а расход снова его достиг). DEBUG терял информацию —
            # пишем INFO с фактическими значениями, но без WARNING-шума.
            logger.info(
                "LLM daily cost limit reached again: $%.2f >= $%.2f — skip "
                "(warning уже был сегодня)",
                current_cost, self._daily_cost_limit,
            )
            self._cost_warning_fresh = False
        return False

    def _log_cost_blocked_pending(self) -> None:
        """
        Диагностика при срабатывании cost-guard: сколько items накопилось pending.

        При хроническом превышении лимита pending растёт неограниченно —
        оператор должен это видеть в логах. Только чтение storage
        (``count_by_state``), никаких изменений в storage/pipeline.

        Диагностика НЕ должна ломать цикл: любые ошибки глушим в DEBUG.
        """
        if not self._storage:
            return
        try:
            pending = self._storage.count_by_state("pending")
        except Exception as e:  # noqa: BLE001 — диагностика не критична
            logger.debug("LLM cost-guard: не удалось получить pending count: %s", e)
            return
        if pending:
            # Minor (лог-шум): при исчерпанном лимите cost-blocked батчей до
            # ~16 за цикл — WARNING даём только первый раз за день (тот же
            # флаг llm_cost_limit_warning_done_YYYYMMDD, что и для самого
            # лимита), повторы — DEBUG.
            if self._cost_warning_fresh:
                logger.warning(
                    "LLM cost-guard: в pending накоплено %d items — они будут "
                    "обработаны после сброса/повышения дневного лимита", pending,
                )
            else:
                logger.debug(
                    "LLM cost-guard: в pending накоплено %d items "
                    "(напоминание уже было сегодня)", pending,
                )

    def _usage_totals(self) -> tuple:
        """
        Возвращает (prompt_tokens, completion_tokens) из usage_stats клиента.

        Используется для вычисления дельты usage одного вызова:
        клиент аккумулирует usage ДО raise (в т.ч. finish_reason=length),
        поэтому дельта учитывает и «сгоревшие» токены.
        """
        stats = getattr(self._client, "usage_stats", None)
        if stats is None:
            return (0, 0)
        try:
            return (
                int(getattr(stats, "prompt_tokens", 0) or 0),
                int(getattr(stats, "completion_tokens", 0) or 0),
            )
        except (TypeError, ValueError):
            return (0, 0)

    def _record_usage(self, before: tuple) -> None:
        """Аккумулирует дневной расход по дельте usage с момента ``before``."""
        if not self._storage:
            return
        after = self._usage_totals()
        # usage_stats может быть сброшен между циклами → отрицательная дельта:
        # в этом случае считаем, что накоплено всё текущее значение.
        prompt_delta = after[0] - before[0]
        completion_delta = after[1] - before[1]
        if prompt_delta < 0:
            prompt_delta = after[0]
        if completion_delta < 0:
            completion_delta = after[1]
        if prompt_delta == 0 and completion_delta == 0:
            return
        self._accumulate_daily_cost(prompt_delta, completion_delta)

    def _accumulate_daily_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """
        Аккумулирует стоимость вызова в дневной счётчик.
        Цена берётся из _pricing по модели.
        """
        if not self._storage:
            return

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        cost_key = f"llm_daily_cost_{today}"

        prices = self._pricing.get(self.model, {})
        input_price = prices.get("input", 0)   # USD per 1M tokens
        output_price = prices.get("output", 0)  # USD per 1M tokens

        # Fallback: если модель не описана в pricing, берём цены,
        # переданные в клиент при создании (input_price/output_price).
        if not input_price and not output_price:
            stats = getattr(self._client, "usage_stats", None)
            input_price = float(getattr(stats, "_input_price", 0) or 0)
            output_price = float(getattr(stats, "_output_price", 0) or 0)

        cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000

        try:
            current = float(self._storage.get_meta(cost_key) or "0")
        except (ValueError, TypeError):
            current = 0.0

        self._storage.set_meta(cost_key, str(current + cost))

    # ── Task 4: length-split recursive batch ───────────────────────────────

    # Максимальная глубина рекурсии для length-split
    _MAX_SPLIT_DEPTH = 5

    def _evaluate_batch_recursive(
        self,
        items: List["NewsItem"],
        feedback_context: Optional[tuple],
        depth: int = 0,
    ) -> List[FilterResult]:
        """
        Рекурсивный evaluate батча с split-ом при length-truncation.

        При finish_reason=length батч разбивается пополам и каждая
        половина ретраится рекурсивно. Результаты сливаются.
        Usage обоих половины учитывается (аккумулируется до raise).

        При depth >= _MAX_SPLIT_DEPTH или len(items)==1 и length →
        возвращает [FilterResult(ok=False) for _ in items] (per-item fallback).

        Parameters
        ----------
        items : List[NewsItem]
        feedback_context : tuple optional
        depth : int — текущая глубина рекурсии

        Returns
        -------
        List[FilterResult]
        """
        if not items:
            return []

        # Terminal: per-item fallback для одиночного item
        if len(items) == 1:
            result = self.evaluate(items[0], feedback_context=feedback_context)
            return [result]

        # Проверка cost guard
        if not self._check_daily_cost():
            self._log_cost_blocked_pending()
            return [FilterResult(ok=False, cost_blocked=True) for _ in items]

        # Собираем user-сообщение
        user_message = build_batch_user_message(items, feedback_context=feedback_context)

        usage_before = self._usage_totals()
        try:
            raw = self._client.chat_json(
                system=BATCH_SYSTEM_PROMPT,
                user=user_message,
            )
        except LLMError as e:
            # usage списан даже при усечении/ошибке ответа — учитываем
            # в дневном лимите до возможного split/retry.
            self._record_usage(usage_before)

            # Minor 3-фикс: основная детекция length-truncation — по ТИПУ
            # исключения (LLMTruncationError), а не по тексту ошибки.
            # Строковый fallback оставлен как запасной (например, если
            # пришёл внешний клиент, бросающий обычный LLMError).
            is_length = isinstance(e, LLMTruncationError)
            if not is_length:
                err_text = str(e).lower()
                is_length = (
                    "finish_reason=length" in err_text
                    or ("length" in err_text and "truncat" in err_text)
                )

            if not is_length:
                # Не length → общий fallback для всех items
                logger.warning(
                    "LLM-ошибка при батч-оценке %d items (non-length): %s — "
                    "fallback для всех: шлём непрофильтрованными",
                    len(items), e,
                )
                return [FilterResult(ok=False) for _ in items]

            # ── length-truncation: рекурсивный split ──────────────────
            if depth >= self._MAX_SPLIT_DEPTH:
                logger.warning(
                    "Length-truncation at depth=%d, %d items — "
                    "per-item fallback (max depth reached)",
                    depth, len(items),
                )
                return [FilterResult(ok=False) for _ in items]

            mid = len(items) // 2
            first_half = items[:mid]
            second_half = items[mid:]

            logger.info(
                "Length-truncation: splitting %d items into "
                "%d + %d (depth %d)",
                len(items), len(first_half), len(second_half), depth + 1,
            )

            # Рекурсивно обрабатываем половины
            # NOTE: feedback_context передаётся в обе половины (once per original batch)
            first_results = self._evaluate_batch_recursive(
                first_half, feedback_context, depth + 1,
            )
            second_results = self._evaluate_batch_recursive(
                second_half, feedback_context, depth + 1,
            )

            return first_results + second_results

        self._record_usage(usage_before)
        return self._parse_batch_response(raw, len(items))

    # ── _build_user_message ─────────────────────────────────────────────────

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

        summary = str(raw.get("summary") or "").strip()
        if not summary:
            logger.warning(
                "LLM вернул пустой summary → fallback: шлём непрофильтрованным"
            )
            return FilterResult(ok=False)

        reason = str(raw.get("reason") or "").strip()

        # ── T2-11: парсинг флага urgent ───────────────────────────────
        # Принимаем: bool, "true"/"false" (case-insensitive), 1/0 (int).
        # При отсутствии или невалидном значении → False (безопасный дефолт).
        raw_urgent = raw.get("urgent")
        urgent: bool = False
        if isinstance(raw_urgent, bool):
            urgent = raw_urgent
        elif isinstance(raw_urgent, int) and not isinstance(raw_urgent, bool):
            urgent = bool(raw_urgent)
        elif isinstance(raw_urgent, str):
            s = raw_urgent.strip().lower()
            if s in ("true", "1", "yes", "on"):
                urgent = True
            elif s in ("false", "0", "no", "off"):
                urgent = False
            # всё остальное (включая пустую строку, None и т.п.) → False

        return FilterResult(
            ok=True,
            score=score,
            summary=summary,
            reason=reason,
            urgent=urgent,
        )

    # ── detect_duplicates ─────────────────────────────────────────────────────────

    # Порог чанкования: не более 40 кандидатов на один LLM-вызов
    _DEDUP_CHUNK_SIZE = 40

    def detect_duplicates(
        self,
        items: List["NewsItem"],
        recent: List[dict],
    ) -> Set[int]:
        """
        Группирует семантические дубликаты среди кандидатов и relative to recent.

        Отдельный лёгкий LLM-вызов (только заголовки) ДО скоринга.
        Критерий: «одно и то же событие, разные слова/язык/числа».
        Примеры: RU-пара (дроны/БПЛА), EN-пара (Pakistan school).

        Полный сбой (``LLMError``) → пустое множество (безопасно, ничего не пропущено).
        Исчерпанный дневной лимит расходов → пустое множество (LLM не вызывается).
        Битые/несуществующие индексы в ответе → игнорируются.
        Если кандидатов < 2 → вызов не делается, возвращается пустое множество.

        Parameters
        ----------
        items : List[NewsItem]
            Кандидаты этого цикла (новые items, прошедшие лексический дедуп).
        recent : List[dict]
            Недавние новости из БД ``(title, source)`` — для кросс-циклового дедупа.
            Каждый dict с ключами ``title`` и ``source``.

        Returns
        -------
        Set[int]
            0-based индексы items, которые являются дубликатами
            БОЛЕЕ РАННЕЙ новости (из основного списка или из recent).
            Пустое множество — ошибка или кандидатов < 2.
        """
        if len(items) < 2:
            return set()

        # MAJOR-фикс: cost-guard ДО любого LLM-вызова. Без него при исчерпанном
        # дневном лимите скоринг блокируется, а дедуп продолжал тратить деньги
        # каждый цикл (вызывается из pipeline до скоринга и в cluster-deferred).
        # Fail-open: пустое множество = «дублей не найдено», ничего не теряется.
        if not self._check_daily_cost():
            logger.info(
                "LLM-дедуп: дневной лимит расходов исчерпан — пропускаем "
                "(безопасно: дублей не найдено)"
            )
            self._log_cost_blocked_pending()
            return set()

        all_dup_indices: Set[int] = set()

        # Чанкование: не более _DEDUP_CHUNK_SIZE кандидатов на вызов
        for chunk_start in range(0, len(items), self._DEDUP_CHUNK_SIZE):
            chunk = items[chunk_start: chunk_start + self._DEDUP_CHUNK_SIZE]

            # Guard на каждой итерации: лимит мог исчерпаться ПОСЛЕ предыдущего
            # чанка (usage предыдущего вызова уже записан). Fail-open: прерываем
            # цикл, сохраняя уже найденные дубли предыдущих чанков.
            if not self._check_daily_cost():
                logger.info(
                    "LLM-дедуп: дневной лимит исчерпан на чанке [%d:%d] из %d "
                    "items — прерываем, сохраняем уже найденные дубли (%d шт.)",
                    chunk_start,
                    chunk_start + len(chunk),
                    len(items),
                    len(all_dup_indices),
                )
                # Minor-фикс: если лимит исчерпался уже ПОСЛЕ предыдущего чанка,
                # _check_daily_cost выше поставил _cost_warning_fresh=True, но
                # pending-напоминание без этого вызова не попадёт в лог: следующий
                # _check_daily_cost уйдёт в else-ветку, сбросит _fresh, а meta-флаг
                # уже "1" → WARNING за день не появится ни разу (только DEBUG).
                self._log_cost_blocked_pending()
                break

            user_message = build_dedup_user_message(chunk, recent)

            usage_before = self._usage_totals()
            try:
                raw = self._client.chat_json(
                    system=DEDUP_SYSTEM_PROMPT,
                    user=user_message,
                )
            except LLMError as e:
                self._record_usage(usage_before)
                logger.warning(
                    "LLM-ошибка при дедупликации чанка [%d:%d] из %d items: %s "
                    "→ прерываем, сохраняем уже найденные дубли (%d шт.)",
                    chunk_start,
                    chunk_start + len(chunk),
                    len(items),
                    e,
                    len(all_dup_indices),
                )
                # m1-фикс: НЕ сбрасываем all_dup_indices — сохраняем находки
                # предыдущих чанков. Токены уже списаны, информация не потеряна.
                break

            self._record_usage(usage_before)

            # Парсим ответ: {"duplicates": [1, 3, ...]}
            dup_indices = self._parse_dedup_response(
                raw, len(chunk), chunk_start,
            )
            all_dup_indices.update(dup_indices)

        logger.debug(
            "LLM-дедуп: %d/%d кандидатов помечены дубликатами",
            len(all_dup_indices), len(items),
        )
        return all_dup_indices

    def _parse_dedup_response(
        self,
        raw: Dict[str, Any],
        chunk_len: int,
        chunk_offset: int,
    ) -> Set[int]:
        """
        Парсит ответ LLM ``{"duplicates": [1, 2, ...]}`` → Set[int] 0-based.

        Битые/несуществующие/вне-диапазона индексы — игнорируются.
        Невалидная структура → пустое множество.

        Minor #4-фикс — защита от «самоссылки» / лидера чанка:
          (a) Guard лидера: первый item чанка (chunk_offset+1 в 1-based) НЕ может
              быть помечен как дубль — у него нет «более раннего» в чанке.
              LLM может вернуть индекс первого item (ошибка модели или спам),
              иначе первый item чанка никогда не выжил бы для других как «ранний».
          (b) Guard пустого чанка: если модель пометила ВСЕ items чанка как дубли
              (например, весь чанк = дубликаты recent-новостей) — оставляем хотя бы
              первый item чанка (он станет «лидером» в next-цикле). Иначе все items
              получат skipped/deferred и информация о событии потеряется.

        Риск: осознанный компромисс. Если событие полностью описано одним
        item из чанка и он помечен как дубль — событие потеряется для пуша.
        Это приемлемо: дедуп рассчитан на пары (A/B), а не на полное
        совпадение всего чанка. Защита (b) гарантирует: минимум 1 item выживает.

        Parameters
        ----------
        raw : dict
            Распарсенный JSON от LLM.
        chunk_len : int
            Количество items в текущем чанке.
        chunk_offset : int
            Стартовый offset оригинального списка (chunk_start).

        Returns
        -------
        Set[int]
            0-based индексы дублей В ПРЕДЕЛАХ ОРИГИНАЛЬНОГО СПИСКА.
        """
        if not isinstance(raw, dict):
            logger.warning("LLM-дедуп вернул не-объект: %s", type(raw).__name__)
            return set()

        raw_list = raw.get("duplicates")
        if not isinstance(raw_list, list):
            logger.warning(
                "LLM-дедуп: 'duplicates' не список (%s) → пропускаем",
                type(raw_list).__name__,
            )
            return set()

        result: Set[int] = set()

        for val in raw_list:
            idx: Optional[int] = None
            # int — напрямую (bool отклоняется как подкласс)
            if isinstance(val, bool):
                continue
            if isinstance(val, int):
                idx = val
            elif isinstance(val, float):
                # Minor 4-фикс: модели часто возвращают 1.0/2.0 — принимаем
                # только конечные целочисленные значения (как в
                # _parse_batch_response для id).
                if math.isfinite(val) and float(int(val)) == val:
                    idx = int(val)
            elif isinstance(val, str):
                # строковые "1", "2" — парсим
                try:
                    idx = int(val.strip())
                except (ValueError, TypeError):
                    continue

            if idx is None:
                continue  # None, NaN/inf, dict и прочее — пропускаем

            # проверяем границы чанка
            if not (1 <= idx <= chunk_len):
                # несуществующий индекс — игнорируем
                continue

            # Guard (a): лидер чанка (idx=1 1-based) не пропускается как дубль,
            # потому что у него нет «более раннего» в пределах этого чанка.
            # LLM может вернуть 1 по ошибке или спаму.
            #
            # Осознанный компромисс: лидер чанка (в т.ч. кросс-цикловый дубль-
            # лидер) не может быть помечен дублем. Это защищает от ложного скипа
            # ценой пропуска кросс-циклового дубля-лидера. Риск принят: выигрыш
            # от защиты (A/B-пары не теряют «ранний» item) превышает редкий
            # пропуск дубля-лидера.
            if idx == 1:
                logger.debug(
                    "LLM-дедуп: модель пометила лидера чанка (1-based idx=1) как "
                    "дубль — игнорируем (защита от самоссылки)",
                )
                continue

            result.add(chunk_offset + (idx - 1))

        return result