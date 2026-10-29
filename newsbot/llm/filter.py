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
from .client import LLMFatalError, _safe_price
from .prompts import (
    SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT, PHASE1_SYSTEM_PROMPT,
    build_feedback_block, build_batch_user_message, sanitize_item_field,
    DEDUP_SYSTEM_PROMPT, build_dedup_user_message, build_dedup_context_message,
    BuildDedupContext,
    MAX_PROMPT_TITLE_CHARS, MAX_PROMPT_SUMMARY_CHARS, LANG_MARKER,
    format_system_prompt, format_batch_system_prompt,
)

logger = logging.getLogger("newsbot.llm.filter")


# ----------------------------------------------------------------------
# Shared helpers (used across all parser methods)
# ----------------------------------------------------------------------


def _coerce_score(raw_score: Any) -> Optional[int]:
    """
    Coerces a raw score value to int in range 0..10.

    Accepts: int (not bool), integer-valued float, str-parseable int.
    Rejects: bool, NaN/inf/float-out-of-range, non-numeric str.
    Returns ``None`` if the value is missing, out of range, or unparseable.

    This is the single source of truth for score validation across ALL
    response parsers (``_parse_response``, ``_parse_batch_response``,
    ``_parse_phase1_response``, ``_parse_phase2_response``).

    Float semantics: only integer-valued floats are accepted (7.0 → 7).
    Non-integer floats (7.6) are rejected (same as in the old parser).
    """
    if isinstance(raw_score, bool):
        # Explicit check before int — bool is subclass of int in Python
        return None
    if isinstance(raw_score, int) and not isinstance(raw_score, bool):
        if 0 <= raw_score <= 10:
            return raw_score
        return None
    if isinstance(raw_score, float):
        if (math.isfinite(raw_score)
                and float(int(raw_score)) == raw_score
                and 0 <= raw_score <= 10):
            return int(raw_score)
        return None
    if isinstance(raw_score, str):
        s = raw_score.strip()
        try:
            candidate = int(s)
            if 0 <= candidate <= 10:
                return candidate
        except (ValueError, TypeError):
            pass
    return None


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
    location: str = ""  # LLM-extracted location, English, canonical
    title_en: Optional[str] = None  # EN title from LLM; None if not provided / fallback
    fatal: bool = False  # True — LLMFatalError (402/401/403), items stay pending


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

        # V7/V3 task: summary length from config (default 240)
        self._summary_max_chars = self._parse_int_field(
            config_llm, "summary_max_chars", 240
        )

        # V3 task: two-phase scoring
        two_phase_cfg = config_llm.get("two_phase", {}) or {}
        if not isinstance(two_phase_cfg, dict):
            logger.warning(
                "llm.two_phase должен быть словарём, получено %s — "
                "two_phase отключён",
                type(two_phase_cfg).__name__,
            )
            two_phase_cfg = {}
        self._two_phase_enabled = self._parse_bool(two_phase_cfg.get("enabled", False))
        self._two_phase_phase2_min_score = self._parse_int_field(
            two_phase_cfg, "phase2_min_score", 4
        )

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
                        "llm.pricing[%r] должен быть словарём {input, output, cache_hit}, "
                        "получено %s — запись пропущена",
                        model_name, type(prices).__name__,
                    )
                    continue
                try:
                    in_price = float(prices.get("input", 0) or 0)
                    out_price = float(prices.get("output", 0) or 0)
                    cache_hit_price = float(prices.get("cache_hit", 0) or 0)
                except (TypeError, ValueError):
                    logger.warning(
                        "llm.pricing[%r] содержит нечисловые цены "
                        "(input=%r, output=%r, cache_hit=%r) — запись пропущена",
                        model_name, prices.get("input"), prices.get("output"),
                        prices.get("cache_hit"),
                    )
                    continue
                # Minor-фикс: NaN/inf-цена (YAML `.nan`) давала cost=nan →
                # персистентный meta "nan" → cost-guard блокировал ВСЕ вызовы
                # до смены дня. Такая запись бесполезна для расчёта — пропускаем
                # её (модель останется без цен → сработает _warn_if_cost_guard_blind).
                # Добавлен cache_hit: при NaN стоимость cache-hit тоже станет nan.
                if not (math.isfinite(in_price) and math.isfinite(out_price)
                        and math.isfinite(cache_hit_price)):
                    logger.warning(
                        "llm.pricing[%r] содержит не конечные цены "
                        "(input=%r, output=%r, cache_hit=%r) — запись пропущена "
                        "(расход по модели не считается)",
                        model_name, prices.get("input"), prices.get("output"),
                        prices.get("cache_hit"),
                    )
                    continue
                self._pricing[model_name] = {
                    "input": in_price,
                    "output": out_price,
                    "cache_hit": cache_hit_price,
                }
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
                max_tokens=self._parse_int_field(config_llm, "max_tokens", 4608),
                input_price=prices.get("input", 0),
                output_price=prices.get("output", 0),
                input_cache_hit_price=prices.get("cache_hit", 0),
            )

        # Minor #1-фикс: если max_tokens не задан в конфиге — используется
        # дефолт 4608 (ранее был 4096, рассинхронизирован с client.py).
        # Предупреждаем, чтобы пользователь явно задал значение в конфиге.
        if "max_tokens" not in config_llm:
            logger.warning(
                "llm.max_tokens не задан в конфиге — используется дефолт 4608. "
                "Для production рекомендуется задать явное значение в config.yaml "
                "(например, max_tokens: 4608 или 8192)."
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
        # Major-фикс: int(float('inf')) бросает OverflowError, int(nan) —
        # ValueError. OverflowError не был перехвачен → LLMFilter падал при
        # старте на `max_tokens: .inf` / `push_threshold: .inf`.
        if isinstance(raw, float) and not math.isfinite(raw):
            logger.warning(
                "llm.%s=%r — не конечное число, используется дефолт %r",
                key, raw, default,
            )
            return default
        try:
            return int(raw)
        except (TypeError, ValueError, OverflowError):
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
            value = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                "llm.%s=%r — ожидалось число, используется дефолт %r",
                key, raw, default,
            )
            return default
        # Minor-фикс: `daily_cost_limit: .nan` проходил как «активный guard»
        # (nan <= 0 == False), но `current < nan` всегда False → ВСЕ вызовы
        # блокировались каждый цикл, items копились в pending. NaN/inf —
        # невалидное конфиг-значение → warning + дефолт.
        if not math.isfinite(value):
            logger.warning(
                "llm.%s=%r — не конечное число, используется дефолт %r",
                key, raw, default,
            )
            return default
        return value

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
        cache_hit_price = prices.get("cache_hit", 0) or 0

        # Minor #3-фикс: если input>0 и output>0, но cache_hit=0 —
        # стоимость cache-hit токенов = 0. Cost-guard будет считать
        # слишком мало, но молча. Warning при старте, если есть hint.
        # Minor #3-фикс: если input/output заданы, но cache_hit нет —
        # cache-hit токены будут стоить 0 → стоимость занижена, но не завышена
        # (безопасный дефолт). Warning при старте.
        if input_price > 0 and output_price > 0 and not cache_hit_price:
            logger.warning(
                "Цена cache_hit не задана в llm.pricing[%r].cache_hit — "
                "стоимость cache-hit токенов будет 0. "
                "Рекомендуется добавить cache_hit в конфиг для точного учёта.",
                self.model,
            )

        if not input_price and not output_price:
            stats = getattr(self._client, "usage_stats", None)
            input_price = _safe_price(getattr(stats, "_input_price", 0) or 0)
            output_price = _safe_price(getattr(stats, "_output_price", 0) or 0)

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

        # V3 task: guard #1 — _check_daily_cost() для прямого вызова evaluate()
        if not self._check_daily_cost():
            self._log_cost_blocked_pending()
            return FilterResult(ok=False, cost_blocked=True)

        user_message = self._build_user_message(item, feedback_context=feedback_context)
        system_prompt = format_system_prompt(self._summary_max_chars)

        usage_before = self._usage_totals()
        try:
            raw = self._client.chat_json(
                system=system_prompt,
                user=user_message,
            )
        except LLMFatalError as e:
            # Major #1-фикс: фатальная ошибка (401/402/403) — НЕ per-item fallback.
            # Items остаются pending для повтора в следующем цикле.
            self._record_usage(usage_before)
            logger.warning(
                "LLM-фатальная ошибка при оценке '%s': %s — "
                "item остаётся pending для повтора в следующем цикле",
                item.title[:60],
                e,
            )
            return FilterResult(ok=False, fatal=True)
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
        Оценивает батч новостей одним LLM-вызовом (V3: two-phase mode).

        V3 two-phase scoring:
          Phase 1 (score-only): compact call returns id/score/urgent for ALL items.
          Phase 2 (full): full call (summary/title_en/location) only for items
            with score >= phase2_min_score (default 4). Items below threshold
            get score from Phase 1 and empty summary/title_en.

        All items numbered 1..N in user-message; LLM returns
        JSON ``{"results": [{"id": 1, ...}, ...]}``.

        V7 fix: ``reason`` field is no longer required in prompts, but parser
        accepts it for backward compatibility.

        Per-item fallback: invalid/missing item → ``ok=False`` for that item.
        Full LLM failure → all items ``ok=False``.

        Parameters
        ----------
        items : List[NewsItem]
        feedback_context : tuple, optional
            (dislikes, likes) — block inserted once per batch.

        Returns
        -------
        List[FilterResult]
            ``len(results) == len(items)``; same order as input.
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
            self._log_cost_blocked_pending()
            return [
                FilterResult(ok=False, cost_blocked=True)
                for _ in items
            ]

        # V3 task: two-phase scoring
        if self._two_phase_enabled and len(items) > 1:
            return self._evaluate_two_phase(items, feedback_context)

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
            score: Optional[int] = _coerce_score(raw_score)

            if score is None:
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
            # Major #3 fix: enforce summary_max_chars even if LLM ignored it.
            if len(summary) > self._summary_max_chars:
                summary = summary[:self._summary_max_chars].rstrip() + "…"
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

            # ── location (Задача: LLM вместо словаря) ──────────────────
            location = str(item_raw.get("location") or "").strip()

            # ── title_en: EN-перевод заголовка от LLM (tolerant) ─────
            # m4-fix: только str. int/float → None (не "42").
            raw_title_en = item_raw.get("title_en")
            title_en: Optional[str] = None
            if isinstance(raw_title_en, str):
                title_en = raw_title_en.strip()
                if not title_en:
                    title_en = None

            results[index] = FilterResult(
                ok=True,
                score=score,
                summary=summary,
                reason=reason,
                urgent=urgent,
                location=location,
                title_en=title_en,
            )

        return results

    # ------------------------------------------------------------------
    # V3 task: two-phase scoring
    # ------------------------------------------------------------------

    def _evaluate_two_phase(
        self,
        items: List["NewsItem"],
        feedback_context: Optional[tuple],
    ) -> List[FilterResult]:
        """
        V3 task: two-phase evaluation.

        Phase 1 (score-only): compact call on ALL items → id/score/urgent.
        Phase 2 (full): full call (summary/title_en/location/urgent) only for
          items where score >= phase2_min_score.

        Cost guard checked before EACH phase (not just before batch).
        If budget exhausted between phases, remaining items get score from
        Phase 1 and empty summary/title_en.

        Parameters
        ----------
        items : List[NewsItem]
        feedback_context : tuple, optional

        Returns
        -------
        List[FilterResult] — same order as input items
        """
        # ── Phase 1: score-only on ALL items ───────────────────────────────
        if not self._check_daily_cost():
            self._log_cost_blocked_pending()
            return [FilterResult(ok=False, cost_blocked=True) for _ in items]

        user_message = build_batch_user_message(items, feedback_context=feedback_context)
        usage_before = self._usage_totals()
        phase1_raw: Optional[Dict[str, Any]] = None
        try:
            phase1_raw = self._client.chat_json(
                system=PHASE1_SYSTEM_PROMPT,
                user=user_message,
            )
        except LLMFatalError as e:
            self._record_usage(usage_before)
            logger.warning(
                "Two-phase: Phase-1 LLMFatalError (score-only): %s — "
                "все items pending",
                e,
            )
            return [FilterResult(ok=False, fatal=True) for _ in items]
        except LLMError as e:
            self._record_usage(usage_before)
            # Major #2 fix: truncation recovery for Phase 1.
            # If the batch was truncated, delegate to _evaluate_batch_recursive
            # which implements split-by-half retry. This restores the original
            # single-path behaviour (replacing the two-phase path entirely).
            is_truncation = isinstance(e, LLMTruncationError)
            if not is_truncation:
                err_text = str(e).lower()
                is_truncation = (
                    "finish_reason=length" in err_text
                    or ("length" in err_text and "truncat" in err_text)
                )
            if is_truncation and self._check_daily_cost():
                logger.info(
                    "Two-phase: Phase-1 truncation — delegating batch of %d items "
                    "to _evaluate_batch_recursive for recovery",
                    len(items),
                )
                return self._evaluate_batch_recursive(items, feedback_context, depth=0)
            logger.warning(
                "Two-phase: Phase-1 LLMError (score-only): %s — "
                "fallback для всех: шлём непрофильтрованными",
                e,
            )
            return [FilterResult(ok=False, cost_blocked=True) for _ in items]

        self._record_usage(usage_before)

        # Parse Phase 1 response — compact: id/score/urgent only
        phase1_results = self._parse_phase1_response(phase1_raw, len(items))

        # ── Identify items that need Phase 2 ──────────────────────────────
        phase2_indices: List[int] = []  # 0-based indices of items needing Phase 2
        for idx, fr in enumerate(phase1_results):
            if fr.ok and fr.score is not None and fr.score >= self._two_phase_phase2_min_score:
                phase2_indices.append(idx)

        if not phase2_indices:
            # All items below threshold — return Phase 1 results as-is
            logger.debug(
                "Two-phase: %d/%d items below phase2_min_score=%d — no Phase 2",
                len(phase1_results), len(items), self._two_phase_phase2_min_score,
            )
            return phase1_results

        # ── Phase 2: full evaluation for passing items ───────────────────
        if not self._check_daily_cost():
            self._log_cost_blocked_pending()
            # Return Phase 1 scores; summary/title_en stay empty
            return phase1_results

        phase2_items = [items[i] for i in phase2_indices]
        phase2_user_message = build_batch_user_message(
            phase2_items, feedback_context=feedback_context,
        )
        usage_before = self._usage_totals()
        phase2_raw: Optional[Dict[str, Any]] = None
        try:
            phase2_raw = self._client.chat_json(
                system=format_batch_system_prompt(self._summary_max_chars),
                user=phase2_user_message,
            )
        except LLMFatalError as e:
            self._record_usage(usage_before)
            logger.warning(
                "Two-phase: Phase-2 LLMFatalError: %s — "
                "items below threshold get phase-1 score only",
                e,
            )
            # Return Phase 1 results (Phase 2 items have empty summary/title_en)
            return phase1_results
        except LLMError as e:
            self._record_usage(usage_before)
            # Major #2 fix: truncation recovery for Phase 2.
            # If the batch was truncated, call _evaluate_batch_recursive on
            # phase-2 items (it splits in half and retries). Merge only the
            # ok=True results back into phase1_results; items that still
            # don't get a valid response fall back to phase-1 data (score/urgent).
            is_truncation = isinstance(e, LLMTruncationError)
            if not is_truncation:
                err_text = str(e).lower()
                is_truncation = (
                    "finish_reason=length" in err_text
                    or ("length" in err_text and "truncat" in err_text)
                )
            if is_truncation and self._check_daily_cost():
                logger.info(
                    "Two-phase: Phase-2 truncation — splitting %d phase-2 items "
                    "via _evaluate_batch_recursive for recovery",
                    len(phase2_items),
                )
                # _evaluate_batch_recursive records its own usage internally;
                # usage_before was already recorded for the failed call above.
                recovered = self._evaluate_batch_recursive(
                    phase2_items, feedback_context, depth=0,
                )
                # Merge recovered results into phase1_results (only ok=True items)
                for p2_idx, rec in enumerate(recovered[:len(phase2_indices)]):
                    if rec.ok:
                        orig_idx = phase2_indices[p2_idx]
                        phase1_results[orig_idx] = FilterResult(
                            ok=True, score=rec.score, summary=rec.summary,
                            urgent=rec.urgent, location=rec.location,
                            title_en=rec.title_en,
                            reason=phase1_results[orig_idx].reason,
                        )
                return phase1_results
            elif is_truncation:
                # Truncation detected but daily budget exhausted — mark phase-2
                # items as cost_blocked so they stay pending and retry tomorrow.
                self._log_cost_blocked_pending()
                for p2_idx in range(len(phase2_items)):
                    orig_idx = phase2_indices[p2_idx]
                    phase1_results[orig_idx] = FilterResult(
                        ok=False,
                        score=phase1_results[orig_idx].score,
                        cost_blocked=True,
                        urgent=phase1_results[orig_idx].urgent,
                    )
                return phase1_results
            logger.warning(
                "Two-phase: Phase-2 LLMError: %s — "
                "phase-2 items get phase-1 score only",
                e,
            )
            return phase1_results

        self._record_usage(usage_before)

        # Parse Phase 2 response: full fields
        phase2_results = self._parse_phase2_response(phase2_raw, len(phase2_items))

        # Merge Phase 2 results back into phase1_results
        phase2_ok_count = 0
        for p2_idx, p2_fr in enumerate(phase2_results):
            orig_idx = phase2_indices[p2_idx]
            if p2_fr.ok:
                phase1_results[orig_idx] = FilterResult(
                    ok=True,
                    score=p2_fr.score,
                    summary=p2_fr.summary,
                    reason=phase1_results[orig_idx].reason,  # keep phase-1 reason if any
                    urgent=p2_fr.urgent,
                    location=p2_fr.location,
                    title_en=p2_fr.title_en,
                )
                phase2_ok_count += 1
            # if not ok — keep phase-1 result (score/urgent from phase-1)

        logger.debug(
            "Two-phase: %d items passed to phase-2, %d phase-2 OK",
            len(phase2_indices), phase2_ok_count,
        )
        return phase1_results

    def _parse_phase1_response(
        self, raw: Dict[str, Any], expected_count: int,
    ) -> List[FilterResult]:
        """
        Parses Phase 1 compact response ``{"results": [{"id":1,"score":7,"urgent":false},...]}``
        → List[FilterResult] with score/urgent filled, summary="" and title_en=None.

        ``reason`` field is ignored (Phase 1 does not return it; parser also
        accepts its absence per V7 requirement).

        Returns ``ok=False`` FilterResult for:
          - non-dict/non-list response structure;
          - invalid/missing id;
          - score outside 0..10;
          - missing ``urgent`` → defaults to False, item stays ok=True
            (changed from earlier draft that promised ``ok=False`` for missing
            urgent; actual behaviour: safe default ``urgent=False`` is applied).
        """
        if not isinstance(raw, dict):
            return [FilterResult(ok=False) for _ in range(expected_count)]

        raw_results = raw.get("results")
        if not isinstance(raw_results, list):
            return [FilterResult(ok=False) for _ in range(expected_count)]

        results: List[FilterResult] = [FilterResult(ok=False) for _ in range(expected_count)]
        seen_ids: Set[int] = set()

        for item_raw in raw_results:
            if not isinstance(item_raw, dict):
                continue

            raw_id = item_raw.get("id")
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

            if parsed_id is None or parsed_id in seen_ids:
                continue
            seen_ids.add(parsed_id)
            index = parsed_id - 1

            # Score
            raw_score = item_raw.get("score")
            score: Optional[int] = _coerce_score(raw_score)

            if score is None:
                results[index] = FilterResult(ok=False)
                continue

            # Urgent
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

            results[index] = FilterResult(
                ok=True,
                score=score,
                summary="",      # Phase 1: no summary
                urgent=urgent,
            )

        return results

    def _parse_phase2_response(
        self, raw: Dict[str, Any], expected_count: int,
    ) -> List[FilterResult]:
        """
        Parses Phase 2 full response (same as _parse_batch_response).

        V7 fix: ``reason`` field is optional — absence does not cause ok=False.
        Backward compatible: if present, saved as reason.
        """
        if not isinstance(raw, dict):
            return [FilterResult(ok=False) for _ in range(expected_count)]

        raw_results = raw.get("results")
        if not isinstance(raw_results, list):
            return [FilterResult(ok=False) for _ in range(expected_count)]

        results: List[FilterResult] = [FilterResult(ok=False) for _ in range(expected_count)]
        seen_ids: Set[int] = set()

        for item_raw in raw_results:
            if not isinstance(item_raw, dict):
                continue

            raw_id = item_raw.get("id")
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

            if parsed_id is None or parsed_id in seen_ids:
                continue
            seen_ids.add(parsed_id)
            index = parsed_id - 1

            # Score
            raw_score = item_raw.get("score")
            score: Optional[int] = _coerce_score(raw_score)

            if score is None:
                results[index] = FilterResult(ok=False)
                continue

            # V7 fix: summary required (Phase 2 contract), but reason is optional
            summary = str(item_raw.get("summary") or "").strip()
            # Major #3 fix: enforce summary_max_chars even if LLM ignored it.
            if len(summary) > self._summary_max_chars:
                summary = summary[:self._summary_max_chars].rstrip() + "…"
            if not summary:
                results[index] = FilterResult(ok=False)
                continue

            # V7 fix: reason is optional (not required in prompt anymore)
            reason = str(item_raw.get("reason") or "").strip()

            # Urgent
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

            # Location
            location = str(item_raw.get("location") or "").strip()

            # title_en
            raw_title_en = item_raw.get("title_en")
            title_en: Optional[str] = None
            if isinstance(raw_title_en, str):
                title_en = raw_title_en.strip()
                if not title_en:
                    title_en = None

            results[index] = FilterResult(
                ok=True,
                score=score,
                summary=summary,
                reason=reason,
                urgent=urgent,
                location=location,
                title_en=title_en,
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

        M11-fix: fail-open при ошибках storage.
        — При ошибке чтения cost/warning-flag → разрешаем вызов LLM
          (худшее: один лишний вызов, не отказ в обслуживании).
        — При ошибке записи warning-flag → WARNING заблокирован,
          но лимит сработает корректно в следующем вызове (cost уже записан).
        — После float() — валидация math.isfinite: NaN в meta → считаем 0.0
          (NaN < limit == False → вечная блокировка до полуночи).
        """
        if self._daily_cost_limit <= 0 or not self._storage:
            return True

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        limit_key = f"llm_daily_cost_{today}"
        warning_flag_key = f"llm_cost_limit_warning_done_{today}"

        # M11-fix: fail-open на чтение + isfinite после float()
        try:
            raw_cost = self._storage.get_meta(limit_key)
            current_cost = float(raw_cost) if raw_cost is not None else 0.0
        except Exception as e:  # noqa: BLE001 — sqlite3.OperationalError и др.
            logger.warning(
                "LLM cost-guard: не удалось прочитать дневной расход (%s) "
                "→ fail-open, вызов разрешён",
                e,
            )
            return True
        if not math.isfinite(current_cost):
            logger.warning(
                "LLM cost-guard: невалидный дневной расход %r (not finite) "
                "→ treated as 0.0",
                raw_cost,
            )
            current_cost = 0.0

        # Решение о блокировке — ВСЕГДА по факту current_cost (не по флагу).
        # Флаг warning'а используется ТОЛЬКО для подавления повторного WARNING:
        # если оператор поднял daily_cost_limit в тот же UTC-день, блокировка
        # снимается сразу, без ожидания полуночи.
        if current_cost < self._daily_cost_limit:
            self._cost_warning_fresh = False
            return True

        # M11-fix: fail-open на чтение warning-flag
        try:
            warning_done = self._storage.get_meta(warning_flag_key)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "LLM cost-guard: не удалось прочитать warning-flag (%s) "
                "→ WARNING заблокирован, но лимит сработает",
                e,
            )
            warning_done = None

        if warning_done != "1":
            logger.warning(
                "LLM daily cost limit reached: $%.2f >= $%.2f — "
                "skipping LLM call (items will be retried later)",
                current_cost, self._daily_cost_limit,
            )
            # M11-fix: fail-open на запись warning-flag
            try:
                self._storage.set_meta(warning_flag_key, "1")
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "LLM cost-guard: не удалось записать warning-flag (%s) "
                    "→ WARNING в этот день будет повторяться",
                    e,
                )
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
        Возвращает (prompt_tokens, completion_tokens, cache_hit, cache_miss)
        из usage_stats клиента.

        Major #12-фикс: возвращаем split cache_hit/cache_miss, чтобы
        _accumulate_daily_cost мог считать стоимость по двум ценам.

        Используется для вычисления дельты usage одного вызова:
        клиент аккумулирует usage ДО raise (в т.ч. finish_reason=length),
        поэтому дельта учитывает и «сгоревшие» токены.
        """
        stats = getattr(self._client, "usage_stats", None)
        if stats is None:
            return (0, 0, 0, 0)
        try:
            return (
                int(getattr(stats, "prompt_tokens", 0) or 0),
                int(getattr(stats, "completion_tokens", 0) or 0),
                int(getattr(stats, "prompt_cache_hit_tokens", 0) or 0),
                int(getattr(stats, "prompt_cache_miss_tokens", 0) or 0),
            )
        except (TypeError, ValueError):
            return (0, 0, 0, 0)

    def _record_usage(self, before: tuple) -> None:
        """Аккумулирует дневной расход по дельте usage с момента ``before``."""
        if not self._storage:
            return
        after = self._usage_totals()
        # before = (prompt, completion, cache_hit, cache_miss)
        # after = (prompt, completion, cache_hit, cache_miss)
        # usage_stats может быть сброшен между циклами → отрицательная дельта:
        # в этом случае считаем, что накоплено всё текущее значение.
        prompt_delta = after[0] - before[0]
        completion_delta = after[1] - before[1]
        cache_hit_delta = after[2] - before[2]
        cache_miss_delta = after[3] - before[3]
        if prompt_delta < 0:
            prompt_delta = after[0]
            cache_hit_delta = after[2]
            cache_miss_delta = after[3]
        if completion_delta < 0:
            completion_delta = after[1]
        if prompt_delta == 0 and completion_delta == 0:
            return
        self._accumulate_daily_cost(prompt_delta, completion_delta,
                                     cache_hit_delta, cache_miss_delta)

    def _accumulate_daily_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_hit_tokens: int = 0,
        cache_miss_tokens: int = 0,
    ) -> None:
        """
        Аккумулирует стоимость вызова в дневной счётчик.
        Цена берётся из _pricing по модели.

        Major #12-фикс: стоимость = hit * price_hit + miss * price_input
        + completion * price_output. При отсутствии split'а
        (cache_hit_tokens == 0 AND cache_miss_tokens == 0) — fallback
        на старую формулу (все prompt по цене input, завышая стоимость).
        Это сохраняет обратную совместимость: старые модели без поддержки
        cache-hit просто получают завышенную оценку (консервативный подход).

        M11-fix: fail-open при ошибках storage.
        — При ошибке чтения текущего cost (get_meta или float()) → выход БЕЗ
          записи: except → return, set_meta НЕ вызывается. БЕЗ этого: код
          продолжился бы к set_meta(..., "0.0") → недоучёт всех предыдущих
          вызовов (аккумулятор перезаписывается нулём при каждой ошибке чтения).
        — После float() — math.isfinite: NaN в meta → current = 0.0.
        — При ошибке записи накопленного cost (set_meta) → no-op (токены уже
          списаны API; дневной лимит сработает позже, в следующих вызовах).
        """
        if not self._storage:
            return

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        cost_key = f"llm_daily_cost_{today}"

        prices = self._pricing.get(self.model, {})
        input_price = prices.get("input", 0)   # USD per 1M tokens
        output_price = prices.get("output", 0)  # USD per 1M tokens
        cache_hit_price = prices.get("cache_hit", 0)  # USD per 1M tokens

        # Fallback: если модель не описана в pricing, берём цены
        # из клиента (input_price/output_price/input_cache_hit_price).
        # _safe_price защищает от битых значений (нечисло, NaN, inf).
        if not input_price and not output_price:
            stats = getattr(self._client, "usage_stats", None)
            input_price = _safe_price(getattr(stats, "_input_price", 0) or 0)
            output_price = _safe_price(getattr(stats, "_output_price", 0) or 0)
            cache_hit_price = _safe_price(getattr(stats, "_input_cache_hit_price", 0) or 0)

        # Major #12-фикс: раздельный расчёт hit/miss.
        # Если нет split'а — fallback на старую формулу (консервативно завышает).
        if cache_hit_tokens == 0 and cache_miss_tokens == 0:
            # Fallback: все prompt_tokens считаем по цене input (завышено, но
            # безопасно — консервативная оценка, не пропускаем дорогие вызовы).
            cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
        else:
            cost = (
                cache_hit_tokens * cache_hit_price
                + cache_miss_tokens * input_price
                + completion_tokens * output_price
            ) / 1_000_000

        # M11-fix: fail-open на чтение + isfinite после float()
        try:
            raw = self._storage.get_meta(cost_key)
            current = float(raw) if raw is not None else 0.0
        except Exception as e:  # noqa: BLE001
            # Выход БЕЗ записи: set_meta не вызывается, накопленный расход не перезаписывается.
            # БЕЗ этого: set_meta(..., "0.0") → недоучёт всех предыдущих вызовов.
            logger.warning(
                "LLM cost-guard: не удалось прочитать дневной расход (%s) "
                "→ накопление пропущено (не перезаписываем аккумулятор)",
                e,
            )
            return
        if not math.isfinite(current):
            logger.warning(
                "LLM cost-guard: невалидный дневной расход %r → current=0.0",
                raw,
            )
            current = 0.0

        # M11-fix: fail-open на запись
        try:
            self._storage.set_meta(cost_key, str(current + cost))
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "LLM cost-guard: не удалось записать накопленный расход (%s) "
                "→ накопление пропущено (токены уже списаны API)",
                e,
            )

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

        # Minor #4-фикс: _check_daily_cost() ДО len(items)==1.
        # Без этого лимит превышается на один вызов evaluate() при батче
        # из 1-го item: evaluate_batch сразу вызывает evaluate() мимо
        # проверки, токены списываются без учёта дневного лимита.
        if not self._check_daily_cost():
            self._log_cost_blocked_pending()
            return [FilterResult(ok=False, cost_blocked=True) for _ in items]

        # Terminal: per-item fallback для одиночного item
        if len(items) == 1:
            result = self.evaluate(items[0], feedback_context=feedback_context)
            return [result]

        # Собираем user-сообщение
        user_message = build_batch_user_message(items, feedback_context=feedback_context)

        usage_before = self._usage_totals()
        try:
            raw = self._client.chat_json(
                system=format_batch_system_prompt(self._summary_max_chars),
                user=user_message,
            )
        except LLMFatalError as e:
            # Major #1-фикс: фатальная ошибка (401/402/403) — items остаются
            # pending. Это отличается от per-item fallback (ok=False): pending
            # означает что цикл попытается снова, когда проблема будет
            # устранена (баланс пополнен, ключ исправлен).
            self._record_usage(usage_before)
            logger.warning(
                "LLM-фатальная ошибка при батч-оценке %d items: %s — "
                "items остаются pending для повтора",
                len(items), e,
            )
            return [FilterResult(ok=False, fatal=True) for _ in items]
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

            # V3 task: guard #2 — check daily cost BEFORE second_half.
            # Usage of first_half is already recorded (before the split).
            # If budget exhausted here, second_half gets cost_blocked results.
            if not self._check_daily_cost():
                self._log_cost_blocked_pending()
                first_results = self._evaluate_batch_recursive(
                    first_half, feedback_context, depth + 1,
                )
                return first_results + [
                    FilterResult(ok=False, cost_blocked=True) for _ in second_half
                ]

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
        score: Optional[int] = _coerce_score(raw_score)

        if score is None:
            logger.warning(
                "LLM вернул score %r (не int 0..10) → fallback: шлём непрофильтрованным",
                raw_score,
            )
            return FilterResult(ok=False)

        summary = str(raw.get("summary") or "").strip()
        # Major #3 fix: enforce summary_max_chars even if LLM ignored it.
        if len(summary) > self._summary_max_chars:
            summary = summary[:self._summary_max_chars].rstrip() + "…"
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

        # ── location (Задача: LLM вместо словаря) ──────────────────────
        # LLM возвращает каноничное название места на английском.
        # Пустая строка / отсутствие → '' (глобальная/непонятная тема).
        location = str(raw.get("location") or "").strip()

        # ── title_en: EN-перевод заголовка от LLM ─────────────────────
        # m4-fix: принимается только str. int/float — вредный выход из LLM
        # (модель ошиблась с форматом) → None (не сохранять "42" как заголовок).
        # Отсутствует/пусто/не-строка → None (не роняем ответ).
        raw_title_en = raw.get("title_en")
        title_en: Optional[str] = None
        if isinstance(raw_title_en, str):
            title_en = raw_title_en.strip()
            if not title_en:
                title_en = None

        return FilterResult(
            ok=True,
            score=score,
            summary=summary,
            reason=reason,
            urgent=urgent,
            location=location,
            title_en=title_en,
        )

    # ── detect_duplicates ─────────────────────────────────────────────────────────

    # Порог чанкования: не более 40 кандидатов на один LLM-вызов
    _DEDUP_CHUNK_SIZE = 40

    # Major #3-фикс: sanity cap на долю помеченных дублей в одном чанке.
    #
    # Почему 0.5 (50%):
    #   — реальный дедуп в news-ленте: 1–3% дублей в одном чанке
    #     (кандидаты уже прошли лексический дедуп, близкие по смыслу пары редки).
    #   — модель может пометить ~30% как дубли при реальном RU/EN-пара событии
    #     (одно и то же событие на разных языках/источниках) — это легитимно.
    #   — 50% — компромисс: пропускаем 30–40% реальных, но отсекаем галлюцинации.
    #   — при срабатывании cap делается переспрос половинами; если и половины
    #     дают >50% — результат чанка отбрасывается (пустой set + warning).
    #     Переспрос выбран вместо немедленного отбрасывания, т.к. 6/10 при 40
    #     источниках по одному событию — легитимный кейс, а отбрасывание всех
    #     40 новостей недопустимо.
    #
    # Минимальный размер для проверки cap — 4 items.
    # При 1–3 items доля дублей может быть 100% (все — дубликаты контекста)
    # и это легитимно. Cap применяется только к чанкам ≥ _DEDUP_SANITY_MIN_CHUNK.
    _DEDUP_SANITY_CAP = 0.5  # 50%

    # Почему 4: при чанке <4 доля дублей 2/3=67% формально превышает cap,
    # но в реальности 2 дубликата из 3 — нормальное событие (ru/en поста
    # об одном и том же происшествии). 4 — минимальный размер, при котором
    # доля 3/4=75% однозначно указывает на галлюцинацию модели.
    _DEDUP_SANITY_MIN_CHUNK = 4  # мин. размер чанка для проверки cap

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

        Cross-chunk dedup: результаты предыдущих чанков (kept leaders + duplicates)
        передаются следующему чанку как CONTEXT. Это позволяет обнаруживать
        кросс-чанковые дубли (например, RU-версия в чанке 1, EN-версия в чанке 2).

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
            БОЛЕЕ РАННЕЙ новости (из основного списка, из recent,
            или из kept leaders предыдущих чанков).
            Пустое множество — ошибка.
        """
        # Guard: пустой список — дедуп невозможен.
        if not items:
            return set()
        # Guard: ровно 1 item — пропускается только при ПУСТОМ RECENT
        # (нет кросс-оконных дублей, лидеру не с чем сравниваться).
        # При непустом RECENT (backfill/guarantee или кросс-оконный дедуп
        # в _cluster_deferred_duplicates) — одиночный кандидат ДОЛЖЕН
        # проверяться против показанных. Pipeline-гарантии сняты;
        # guard здесь блокировал бы ЛЕГИТИМНЫЙ дедуп.
        if len(items) == 1 and not recent:
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
        # Накапливаемый контекст предыдущих чанков (kept leaders + dup items).
        # Каждый следующий чанк видит ВСЕ результаты предыдущих чанков.
        all_ctx: BuildDedupContext = BuildDedupContext(kept=[], dup=[])

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

            user_message = build_dedup_user_message(chunk, recent, ctx=all_ctx)

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
            # M2-fix: лидер может быть помечен, инвариант проверяется выше по стеку.
            dup_indices = self._parse_dedup_response(
                raw, len(chunk), chunk_start,
                recent=recent,
            )

            # Major #3-фикс: если cap превышен (>50%) — _parse_dedup_response
            # вернул None. Переспрашиваем halves чанка, а не отбрасываем.
            # NOTE: при retry halves контекст (all_ctx) теряется для этой части чанка —
            # это компромисс: не можем накопить контекст при retry.
            if dup_indices is None:
                mid = len(chunk) // 2
                first_half = chunk[:mid]
                second_half = chunk[mid:]

                logger.info(
                    "LLM-дедуп: cap превышен на чанке [%d:%d], "
                    "переспрашиваем halves %d+%d",
                    chunk_start, chunk_start + len(chunk),
                    len(first_half), len(second_half),
                )

                # При retry halves теряем кросс-половинный контекст
                # (чанки видят только свои kept-items). Приемлемо.

                # CRITICAL-фикс: проверяем лимит ПЕРЕД halves; при исчерпании —
                # fail-open (пустое множество), как гарантирует docstring.
                if not self._check_daily_cost():
                    logger.info(
                        "LLM-дедуп: дневной лимит исчерпан на half-retry чанка "
                        "[%d:%d] — fail-open, дубли не найдены",
                        chunk_start, chunk_start + len(chunk),
                    )
                    self._log_cost_blocked_pending()
                    chunk_dup_set = set()
                else:
                    first_half_dups = set()
                    if first_half:
                        usage_before = self._usage_totals()
                        try:
                            first_raw = self._client.chat_json(
                                system=DEDUP_SYSTEM_PROMPT,
                                user=build_dedup_user_message(
                                    first_half, recent, ctx=all_ctx,
                                ),
                            )
                        except LLMError as e:
                            self._record_usage(usage_before)
                            logger.warning(
                                "LLM-ошибка при half-retry[0:%d] из [%d:%d]: %s "
                                "→ результат игнорируется для этой половины",
                                len(first_half),
                                chunk_start, chunk_start + len(chunk),
                                e,
                            )
                        else:
                            self._record_usage(usage_before)
                            first_parsed = self._parse_dedup_response(
                                first_raw, len(first_half), 0,
                                recent=None,
                            )
                            if first_parsed is None:
                                logger.warning(
                                    "LLM-дедуп: halves[0:%d] тоже превысили cap — "
                                    "результат игнорируется для этой половины",
                                    len(first_half),
                                )
                            else:
                                first_half_dups = first_parsed
                                # CRITICAL-фикс: LLM возвращает 1-based внутри half;
                                # конвертируем через chunk_start (абсолютный offset).
                                all_dup_indices.update(
                                    chunk_start + idx for idx in first_half_dups
                                )
                                for local_idx in range(len(first_half)):
                                    abs_idx = chunk_start + local_idx
                                    item = items[abs_idx]
                                    if local_idx in first_half_dups:
                                        all_ctx.dup.append({
                                            "id": abs_idx,
                                            "title": item.title,
                                            "source": getattr(item, "source", ""),
                                        })
                                    else:
                                        all_ctx.kept.append({
                                            "id": abs_idx,
                                            "title": item.title,
                                            "source": getattr(item, "source", ""),
                                        })

                    second_half_dups = set()
                    if second_half:
                        usage_before = self._usage_totals()
                        try:
                            second_raw = self._client.chat_json(
                                system=DEDUP_SYSTEM_PROMPT,
                                user=build_dedup_user_message(
                                    second_half, recent, ctx=all_ctx,
                                ),
                            )
                        except LLMError as e:
                            self._record_usage(usage_before)
                            logger.warning(
                                "LLM-ошибка при half-retry[%d:%d] из [%d:%d]: %s "
                                "→ результат игнорируется для этой половины",
                                chunk_start + len(first_half),
                                chunk_start + len(chunk),
                                chunk_start, chunk_start + len(chunk),
                                e,
                            )
                        else:
                            self._record_usage(usage_before)
                            second_parsed = self._parse_dedup_response(
                                second_raw, len(second_half), 0,
                                recent=None,
                            )
                            if second_parsed is None:
                                logger.warning(
                                    "LLM-дедуп: halves[%d:%d] тоже превысили cap — "
                                    "результат игнорируется для этой половины",
                                    chunk_start + len(first_half),
                                    chunk_start + len(chunk),
                                )
                            else:
                                second_half_dups = second_parsed
                                # CRITICAL-фикс: LLM возвращает 1-based внутри half;
                                # конвертируем через chunk_start+len(first_half).
                                second_half_offset = chunk_start + len(first_half)
                                all_dup_indices.update(
                                    second_half_offset + idx
                                    for idx in second_half_dups
                                )
                                for local_idx in range(len(second_half)):
                                    abs_idx = second_half_offset + local_idx
                                    item = items[abs_idx]
                                    if local_idx in second_half_dups:
                                        all_ctx.dup.append({
                                            "id": abs_idx,
                                            "title": item.title,
                                            "source": getattr(item, "source", ""),
                                        })
                                    else:
                                        all_ctx.kept.append({
                                            "id": abs_idx,
                                            "title": item.title,
                                            "source": getattr(item, "source", ""),
                                        })

                    # CRITICAL-фикс: halves записали в all_ctx напрямую;
                    # пропускаем общий цикл ниже, чтобы не было двойного append.
                    chunk_dup_set = set()
                    continue  # ← CRITICAL: не дать общему циклу дописать items второй раз
            else:
                all_dup_indices.update(dup_indices)
                chunk_dup_set = dup_indices  # Set[int] 0-based within full list

                # ── Cross-chunk context accumulation ─────────────────────────────────
                # После успешной обработки чанка накапливаем его результаты
                # в контексте для следующих чанков:
                #   - kept: все non-dup items чанка → лидеры для будущих чанков
                #   - dup: все dup items чанка с привязкой к их kept-оригиналам
                #     (в пределах того же чанка — лидеры не в all_dup_indices)

            # MAJOR 1-фикс: ведём абсолютный индекс по items (не внутренний
            # итератор range). Ключевая ошибка была: items[item_idx - chunk_start]
            # при item_idx=chunk_start+1 для чанка 2 давало items[1] из чанка 1
            # (всего чанк=2, items уже содержит данные чанка 1 → неправильное
            # совпадение); для чанка 3+ items[:chunk_size] не содержал предыдущие
            # чанки → контекст был пуст.
            # Фикс: используем abs_idx (= item_idx) как абсолютный индекс в items,
            # а chunk_local = abs_idx - chunk_start для доступа к chunk[].
            for abs_idx in range(chunk_start, chunk_start + len(chunk)):
                item = items[abs_idx]  # MAJOR 1-фикс: абсолютный индекс в items

                if abs_idx in chunk_dup_set:
                    # Дубль: привязываем к kept-лидеру (MAJOR 1-фикс: C/D numbering).
                    # Вариант «без «duplicate of Cx»» — храним только title/source
                    # дубля для контекста, LLM сопоставит по заголовку итак,
                    # а разъехавшиеся номера C при наличии пропусков между чанками
                    # больше невозможны (C-нумерация сквозная, enumerate(start=1)
                    # во всех чанках, а kept.id — абсолютный; без привязки
                    # «duplicate of Cx» контекст остаётся корректным).
                    all_ctx.dup.append({
                        "id": abs_idx,
                        "title": item.title,
                        "source": getattr(item, "source", ""),
                        # MAJOR 3-фикс: summary/url не выводятся в промпт
                        # (build_dedup_context_message использует только title+source),
                        # хранить их в all_ctx нет смысла — экономия памяти при N→∞.
                        # original_id убран: без «duplicate of Cx» в промпте
                        # привязка не нужна, а её вычисление ломалось на кросс-чанках.
                    })
                else:
                    # Kept item — лидер для будущих чанков
                    all_ctx.kept.append({
                        "id": abs_idx,
                        "title": item.title,
                        "source": getattr(item, "source", ""),
                        # MAJOR 3-фикс: summary/url не выводятся в промпт.
                    })

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
        recent: Optional[List[dict]] = None,
    ) -> Optional[Set[int]]:
        """
        Парсит ответ LLM ``{"duplicates": [1, 2, ...]}`` → Set[int] 0-based.

        Битые/несуществующие/вне-диапазона индексы — игнорируются.
        Невалидная структура → пустое множество.

        Major #3-фикс: при срабатывании sanity-cap (>50%) возвращает None,
        чтобы вызывающая сторона (``detect_duplicates``) могла переспросить
        halves, а не отбрасывать результат сразу.

        CRITICAL-фикс: cap НЕ применяется при непустом ``recent`` — в этом
        режиме 100% дублей легитимны (item дублирует уже показанную новость,
        pipeline.py:1413-1417), cap отбрасывал бы обе halves → пустой результат.

        M2-fix: ЛЮБОЙ item (включая лидера чанка, idx=1) может быть помечен
        как дубль более нового/полного item'а из того же чанка — это валидное
        поведение (правило "newer becomes leader" в DEDUP_SYSTEM_PROMPT :537-543).
        Инвариант «хотя бы один item остаётся» обеспечивается вызывающей
        стороной: для digest-пути — pipeline.py:1362, для ingest-пути —
        pipeline.py (Major ingest safety, _run_llm_dedup).

        Parameters
        ----------
        raw : dict
            Распарсенный JSON от LLM.
        chunk_len : int
            Количество items в текущем чанке.
        chunk_offset : int
            Стартовый offset оригинального списка (chunk_start).
        recent : list of dict, optional
            Недавние новости из БД. При непустом списке cap не применяется
            (все дубли легитимны).

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

            # M2-fix: leader of first chunk (idx=1) may be marked as duplicate
            # of a newer/more-complete item from the same chunk -- valid behavior
            # (rule "newer becomes leader" in DEDUP_SYSTEM_PROMPT :537-539).
            # Invariant "at least one item remains" is guaranteed by caller:
            # digest path — pipeline.py:1362; ingest path — pipeline.py
            # (Major ingest safety, _run_llm_dedup).

            result.add(chunk_offset + (idx - 1))

        # Major #3-фикс: sanity cap — если модель пометила более 50% чанка
        # как дубли, это галлюцинация. Возвращаем None, чтобы вызывающая
        # сторона (detect_duplicates) могла переспросить halves, а не
        # отбрасывать результат сразу. Инвариант «хотя бы один остаётся»
        # сохраняется (cap > 50%).
        #
        # Cap применяется только к чанкам ≥ _DEDUP_SANITY_MIN_CHUNK (4 items).
        # При маленьких чанках доля дублей может быть высокой по легитимным
        # причинам (3 из 3 — все дубли контекста), cap бы ложно срабатывал.
        #
        # CRITICAL-фикс: cap НЕ применяется при непустом ``recent`` — все
        # дубли в этом режиме легитимны (pipeline.py:1413-1417).
        if (chunk_len >= self._DEDUP_SANITY_MIN_CHUNK
                and recent is not None and not recent):  # cap когда recent ЕСТЬ но ПУСТОЙ
            dup_fraction = len(result) / chunk_len
            if dup_fraction > self._DEDUP_SANITY_CAP:
                logger.warning(
                    "LLM-дедуп: модель пометила %d/%d (%.0f%%) кандидатов чанка "
                    "как дубли — превышен sanity-cap (%.0f%%). "
                    "Результат игнорируется, items остаются pending, "
                    "переспрашиваем halves…",
                    len(result), chunk_len, dup_fraction * 100,
                    self._DEDUP_SANITY_CAP * 100,
                )
                return None  # Major #3: передаём управление в detect_duplicates

        return result
