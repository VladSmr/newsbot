"""
DeepSeek API-клиент (OpenAI-compatible).
Все сетевые и парсинговые ошибки оборачиваются в LLMError или
корректный "пустой" BalanceInfo — вызывающая сторона не падает.
"""
import json
import logging
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger("newsbot.llm.client")


# ─────────────────────────────────────────────────────────────────
# Usage accumulator
# ─────────────────────────────────────────────────────────────────


def _safe_int(value: Any, default: int = 0) -> int:
    """Безопасное преобразование в int. При ошибке — default.

    Major-фикс: usage-поля из ответа API могут быть Infinity/NaN (битый или
    нестандартный ответ). ``int(float('inf'))`` бросает OverflowError, который
    не перехватывался → LLMError → валидный content выбрасывался. Usage —
    вспомогательные данные (cost-guard), они не должны ронять разбор ответа:
    невалидное число → default.
    """
    if value is None:
        return default
    if isinstance(value, float) and not math.isfinite(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _reject_json_constant(value: str) -> Any:
    """parse_constant для json.loads: NaN/Infinity/-Infinity не валидный JSON.

    Стандартный json.loads принимает эти литералы (в т.ч. `{"x": NaN}`),
    хотя RFC 8259 их запрещает. Цель — не принять битый/нестандартный ответ
    модели, поэтому бросаем ValueError (перехватывается ниже, уходит в ветку
    невалидного JSON).
    """
    raise ValueError(f"недопустимая JSON-константа: {value}")


def _parse_cached_tokens(usage: Dict[str, Any]) -> Tuple[int, int]:
    """
    Извлекает cached_hit и cached_miss из usage-блока.

    Поддерживает три формы ответа:
      1) OpenAI-стиль: usage["prompt_tokens_details"]["cached_tokens"]
         → cached_hit = cached_tokens,  cached_miss = prompt - cached_hit
      2) DeepSeek top-level: usage["prompt_cache_hit_tokens"] /
                              usage["prompt_cache_miss_tokens"]
         → берём оба поля напрямую
      3) Fallback: только cached_hit, cached_miss = prompt - cached_hit (не < 0)

    Returns:
        (cached_hit, cached_miss)
    """
    prompt_tokens = _safe_int(usage.get("prompt_tokens"), 0)

    # 1) OpenAI-стиль: prompt_tokens_details.cached_tokens
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and "cached_tokens" in details:
        cached_hit = _safe_int(details["cached_tokens"], 0)
        cached_miss = max(0, prompt_tokens - cached_hit)
        return cached_hit, cached_miss

    # 2) DeepSeek top-level: оба поля
    cached_hit_ds = _safe_int(usage.get("prompt_cache_hit_tokens"), None)
    cached_miss_ds = _safe_int(usage.get("prompt_cache_miss_tokens"), None)

    if cached_hit_ds is not None:
        if cached_miss_ds is not None:
            # оба DeepSeek-поля присутствуют
            return cached_hit_ds, cached_miss_ds
        else:
            # есть только cached_hit → miss = остаток
            return cached_hit_ds, max(0, prompt_tokens - cached_hit_ds)

    # 3) ни один формат не обнаружен
    return 0, max(0, prompt_tokens)


@dataclass
class UsageStats:
    """
    Аккумулятор LLM-usage за цикл. Потокобезопасность не требуется
    (pipeline — однопоточный).
    """

    model: str = "unknown"
    calls: int = 0
    prompt_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Цены из конфига (USD за 1M токенов)
    _input_price: float = 0.0
    _output_price: float = 0.0
    _input_cache_hit_price: float = 0.0  # цена cache-hit (DeepSeek off-peak: $0.003/M)

    def set_prices(self, input_price: float, output_price: float,
                   input_cache_hit_price: float = 0.0) -> None:
        """Устанавливает цены модели (USD за 1M токенов)."""
        self._input_price = input_price
        self._output_price = output_price
        self._input_cache_hit_price = input_cache_hit_price

    @property
    def total_cost_usd(self) -> float:
        """
        Вычисляет общую стоимость всех вызовов в USD.

        Major #12-фикс: стоимость = cache_hit_tokens * price_cache_hit
                             + cache_miss_tokens * price_input
                             + completion_tokens * price_output
        Все делится на 1_000_000.

        При отсутствии split'а hit/miss (prompt_cache_hit_tokens == 0 и
        prompt_cache_miss_tokens == 0) — fallback на старую формулу
        (все prompt_tokens по цене input, без разделения).
        """
        if (self.prompt_cache_hit_tokens == 0
                and self.prompt_cache_miss_tokens == 0):
            # Fallback: нет split'а hit/miss → вся сумма по цене cache_miss
            # (старое поведение — завышает стоимость, но сохраняет контракт).
            cost = (
                self.prompt_tokens * self._input_price
                + self.completion_tokens * self._output_price
            ) / 1_000_000
        else:
            # Новый расчёт: раздельный учёт hit и miss.
            cost = (
                self.prompt_cache_hit_tokens * self._input_cache_hit_price
                + self.prompt_cache_miss_tokens * self._input_price
                + self.completion_tokens * self._output_price
            ) / 1_000_000
        return cost

    def add_response(self, data: Dict[str, Any]) -> None:
        """Добавляет usage из ответа API (data — распарсенный JSON). Безопасно при отсутствии."""
        self.calls += 1

        usage: Any = data.get("usage")
        if not isinstance(usage, dict):
            return

        self.total_tokens += max(0, _safe_int(usage.get("total_tokens"), 0))
        self.completion_tokens += max(0, _safe_int(usage.get("completion_tokens"), 0))

        prompt_tokens = _safe_int(usage.get("prompt_tokens"), 0)
        self.prompt_tokens += prompt_tokens

        cached_hit, cached_miss = _parse_cached_tokens(usage)
        self.prompt_cache_hit_tokens += cached_hit
        self.prompt_cache_miss_tokens += cached_miss

    def log_cycle(self) -> None:
        """Логирует одну агрегированную строку на уровне INFO."""
        if self.calls == 0:
            return
        logger.info(
            "LLM за цикл: calls=%d, prompt=%d (cached=%d, miss=%d), "
            "completion=%d, total=%d",
            self.calls, self.prompt_tokens,
            self.prompt_cache_hit_tokens, self.prompt_cache_miss_tokens,
            self.completion_tokens, self.total_tokens,
        )

    def reset(self) -> None:
        """Сбрасывает счётчики (вызывается в начале нового цикла)."""
        self.calls = 0
        self.prompt_tokens = 0
        self.prompt_cache_hit_tokens = 0
        self.prompt_cache_miss_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        # Цены сохраняются при reset


class LLMError(Exception):
    """Единая ошибка LLM-слоя: любая неудача при вызове API или парсинге ответа."""

    pass


class LLMTruncationError(LLMError):
    """
    Ответ LLM усечён по max_tokens (finish_reason=length).

    Отдельный подкласс LLMError, чтобы вызывающая сторона (llm/filter.py)
    могла отличить усечение от прочих ошибок по ТИПУ исключения,
    а не по разбору текста сообщения.
    Наследуется от LLMError — существующие `except LLMError` продолжают
    ловить эту ошибку без изменений.
    """

    pass


class LLMFatalError(LLMError):
    """
    Фатальная ошибка LLM — вызвана неисправимой HTTP-ошибкой 4xx
    (кроме 429). Это означает:
      — 401/403: проблема с авторизацией (ключ, права, ban);
      — 402: недостаточно средств на аккаунте;
    При таких ошибках НЕ имеет смысла делать retry — ситуация
    не изменится сама по себе (баланс не пополнится, ключ не станет
    валидным). Отличие от LLMError важно для вызывающей стороны:
    при LLMFatalError НЕ нужно помечать все items батча как ok=False
    (per-item fallback), достаточно вернуть их в pending для повтора
    в следующем цикле, когда проблема будет устранена.

    Наследуется от LLMError — существующие ``except LLMError`` продолжают
    ловить без изменений. Новый код может ловить конкретный подкласс,
    чтобы отличать фатальную ошибку от транзиентной.
    """

    pass


class LLMContextLengthError(LLMTruncationError):
    """
    Контекстная длина превышена (HTTP 400 или finish_reason=length).

    Ключевое отличие от LLMTruncationError: проблема связана с размером
    входного промпта, а не ограничением вывода. Фильтр при получении
    этого исключения знает, что нужно делать split по количеству items,
    а не per-item fallback.

    Наследуется от LLMTruncationError → LLMError — обратная совместимость
    сохранена.
    """

    pass


class BalanceInfo:
    """Информация о балансе аккаунта."""

    __slots__ = ("is_available", "currency", "total_balance")

    def __init__(self, is_available: bool, currency: str, total_balance: float):
        self.is_available = is_available
        self.currency = currency
        self.total_balance = total_balance

    def __repr__(self) -> str:
        return (
            f"BalanceInfo(is_available={self.is_available}, "
            f"currency={self.currency!r}, total_balance={self.total_balance})"
        )


# ─────────────────────────────────────────────────────────────────
# JSON parsing helpers (Major #2 fix)
# ─────────────────────────────────────────────────────────────────


_JSON_FENCE_RE = re.compile(
    r"^\s*(?:```(?:json)?\s*\n?|Here(?:'s| is) (?:the )?JSON:?\s*)"  # opening fences / preamble
    r"|"
    r"\n?\s*```\s*$",  # closing fences
    re.IGNORECASE | re.MULTILINE
)

# NOTE: _BRACE_RE and _LAST_BRACE_RE were removed (dead code, unused).


def _strip_json_fences(text: str) -> str:
    """
    Major #2-фикс: удаляет markdown-обрамления и преамбулу из LLM-ответа
    перед передачей в json.loads.

    Убирает (в порядке приоритета):
      1. ```json … ``` или ``` … ``` (в т.ч. с языковым тегом).
      2. Here is the JSON: {…} / Here is JSON: {…} / Here's JSON: {…}.
      3. Here is the response: {…} и аналогичные варианты.
      4. При неудаче — извлечение JSON по позициям '{' через
         ``json.JSONDecoder().raw_decode`` (корректный учёт строк:
         ``{"s": "a } b"}`` не ломается).

    Returns
    -------
    str
        Текст, очищенный от обрамлений. Пустая строка для пустого/None входа.
    """
    if not text:
        return ""

    stripped = _JSON_FENCE_RE.sub("", text).strip()

    # Если после снятия обрамления осталось что-то валидное (начинается с '{')
    # — возвращаем. Иначе пытаемся вытащить JSON из середины текста.
    if stripped.startswith("{"):
        return stripped

    # Rescue: используем json.JSONDecoder().raw_decode по позициям '{',
    # который корректно учитывает строки (экранированные кавычки,
    # кавычки внутри строк — ``{"s": "a } b"}`` не ломается).
    try:
        decoder = json.JSONDecoder()
        brace_start = text.find("{")
        if brace_start != -1:
            candidate, _ = decoder.raw_decode(text[brace_start:])
            return json.dumps(candidate)  # serialize back to string for caller
    except (json.JSONDecodeError, ValueError):
        pass

    # Ничего не помогло — возвращаем что есть (json.loads сам выдаст ошибку)
    return stripped

def _safe_price(value: Any, default: float = 0.0) -> float:
    """
    Minor #1-фикс: безопасное преобразование цены в float.

    При ошибке — default + warning в лог. Защищает от сбоя при
    ``float(getattr(stats, "_input_price", 0) or 0)`` когда значение —
    нечисловая строка или объект без __float__.
    """
    try:
        result = float(value)
        if not math.isfinite(result):
            logger.warning(
                "Невалидная цена %r (не конечное число) → используется %.4f",
                value, default,
            )
            return default
        return result
    except (TypeError, ValueError):
        logger.warning(
            "Невалидная цена %r (type=%s) → используется %.4f",
            value, type(value).__name__, default,
        )
        return default


# ─────────────────────────────────────────────────────────────────
# DeepSeekClient
# ─────────────────────────────────────────────────────────────────


class DeepSeekClient:
    """
    Клиент для DeepSeek API (OpenAI-compatible, /chat/completions).

    Параметры
    ----------
    api_key : str
    base_url : str   — например https://api.deepseek.com
    model : str      — имя модели, например deepseek-flash
    timeout : float — таймаут одного запроса, сек
    max_retries : int — максимум повторов при 429/5xx/транзиентных ошибках
    temperature : float — temperature для chat-запроса (default 0)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 4608,
        input_price: float = 0.0,
        output_price: float = 0.0,
        input_cache_hit_price: float = 0.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Major #13-фикс: max_retries=0 и отрицательные нормализуются в 1 попытку.
        # Явный warning, чтобы оператор понимал, что конфиг изменён.
        if max_retries <= 0:
            logger.warning(
                "max_retries=%d — трактуется как 1 попытка. "
                "LLM будет вызван ровно один раз (до первой ошибки). "
                "Если нужны повторы — задайте max_retries: 1 или больше.",
                max_retries,
            )
            max_retries = 1
        self.max_retries = max_retries
        self.temperature = temperature
        # max_tokens: лимит на completion. Батч (≤10 items, ~100-200 токенов/item)
        # требует больше, чем одиночная оценка (~150-300 токенов).
        # Minor #6-фикс: дефолт 4608 (ранее 4096; конфиг позволял 4608).
        # При batch_size=10: ≈400-600 токенов/item × 10 = 4000-6000 токенов ответ,
        # модель обычно укладывается в 4600, запас до 4608 покрывает типичный
        # батч 10 items + погрешность модели. Конфиг позволяет переопределить.
        self.max_tokens = max_tokens
        # один HTTP-клиент на весь жизненный цикл — переиспользуем соединения
        self._http = httpx.Client(timeout=self.timeout, trust_env=True)  # подхватывает HTTP_PROXY / HTTPS_PROXY из окружения (Docker)
        self.usage_stats = UsageStats(model=model)
        # Устанавливаем цены для расчёта стоимости (включая cache-hit)
        self.usage_stats.set_prices(input_price, output_price, input_cache_hit_price)

    def close(self) -> None:
        """Закрывает HTTP-клиент. Идемпотентен."""
        if self._http is not None:
            self._http.close()
            self._http = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_json(self, system: str, user: str) -> Dict[str, Any]:
        """
        Отправляет chat-запрос и возвращает распарсенный JSON из ответа.

        Параметры
        ----------
        system : str — system-prompt
        user   : str — user message

        Возвращает
        -----------
        dict — содержимое поля "content" из assistant message,
               распарсенное как JSON.

        Исключения
        -----------
        LLMError — при неудаче после всех попыток или некорректном ответе.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

        last_exc: Optional[Exception] = None
        # Minor #3-фикс: ровно 1 retry на JSONDecodeError (не бесконечный цикл).
        # Флаг _json_retry_done устанавливается при первом JSONDecodeError и
        # блокирует второй retry; не сбрасывается между попытками — сброс
        # после успешного ответа невозможен (return выходит из функции).
        _json_retry_done = False

        for attempt in range(1, max(1, self.max_retries) + 1):
            try:
                response = self._post("/chat/completions", payload)
                data: Any = response.json()

                # Защита: top-level должен быть dict
                if not isinstance(data, dict):
                    raise LLMError(
                        f"Некорректный формат ответа API: ожидался dict, "
                        f"получен {type(data).__name__}"
                    )

                # MINOR-фикс: аккумулируем usage СРАЗУ после проверки «data — dict».
                # Токены уже списаны провайдером на любой ветке (в т.ч. raise на
                # choices/message/пустом content/finish_reason=length), поэтому
                # учёт делается ровно один раз на полученный response.
                # Ниже дублирующих вызовов НЕТ (идемпотентность по построению).
                self._log_usage_debug(data)

                choices: Any = data.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise LLMError("Пустой или отсутствующий массив choices в ответе API")

                first_choice: Any = choices[0]
                if not isinstance(first_choice, dict):
                    raise LLMError(f"choices[0] не dict: {type(first_choice).__name__}")

                message: Any = first_choice.get("message")
                if not isinstance(message, dict):
                    raise LLMError(f"message не dict: {type(message).__name__}")

                content = message.get("content", "")
                if not content:
                    raise LLMError("Пустой content в assistant message")

                # Major #2-фикс / P5: finish_reason=length сам по себе НЕ значит,
                # что JSON битый — валидный закрытый JSON может уложиться ровно
                # в max_tokens на последнем токене. Поэтому сначала пробуем
                # json.loads: успех → принимаем ответ (иначе split → повторный
                # прогон тех же items → двойная оплата). LLMTruncationError —
                # только если парсинг действительно не удался.
                finish_reason: Any = first_choice.get("finish_reason")

                # ── usage-логирование уже выполнено выше (одно на response) ──
                # Major #2-фикс: снимаем markdown-обрамление и преамбулу
                # перед json.loads. Модель может вернуть ```json\n{…}\n``` или
                # «Here is the JSON: {…}» — оба случая теперь корректно парсятся.
                try:
                    result: Any = json.loads(
                        _strip_json_fences(content), parse_constant=_reject_json_constant
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    if finish_reason == "length":
                        # usage уже аккумулирован выше (сразу после проверки dict) —
                        # токены учтены даже при length-truncation.
                        logger.warning(
                            "LLM ответ усечён (finish_reason=length, max_tokens=%d) "
                            "и JSON не парсится — fallback для батча: "
                            "шлём непрофильтрованными",
                            self.max_tokens,
                        )
                        raise LLMTruncationError(
                            f"LLM-ответ усечён (finish_reason=length, "
                            f"max_tokens={self.max_tokens}). "
                            f"JSON может быть неполным — fallback "
                            f"(ok=False для всех items батча)."
                        )
                    raise

                if finish_reason == "length":
                    logger.debug(
                        "LLM ответ с finish_reason=length, но JSON валиден — "
                        "принимаем без fallback (max_tokens=%d)",
                        self.max_tokens,
                    )

                # Модель может вернуть массив или примитив
                if not isinstance(result, dict):
                    raise LLMError(
                        f"Модель вернула не-объект: ожидался dict, "
                        f"получен {type(result).__name__}"
                    )
                return result

            # ── транзиентные ошибки — ретраим ────────────────────────

            except httpx.TimeoutException as e:
                last_exc = e
                logger.warning(
                    "Попытка %d/%d: таймаут при обращении к LLM API",
                    attempt,
                    self.max_retries,
                )

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 429:
                    retry_after = e.response.headers.get("retry-after")
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except (ValueError, TypeError):
                            logger.warning(
                                "Невалидный Retry-After %r — используется backoff %d сек",
                                retry_after, 2 ** attempt,
                            )
                            wait = 2 ** attempt
                        else:
                            if not math.isfinite(wait) or wait <= 0:
                                logger.warning(
                                    "Retry-After %r вне допустимого диапазона — "
                                    "используется backoff %d сек",
                                    retry_after, 2 ** attempt,
                                )
                                wait = 2 ** attempt
                    else:
                        wait = 2 ** attempt
                    # Major #1-фикс: cap wait на [1, 120] сек.
                    # max() гарантирует минимум 1 сек (защита от time.sleep(0)
                    # и edge-case отрицательных значений). min() ограничивает
                    # верхнюю границу (как в telegram-слое).
                    wait = min(max(wait, 1), 120)
                    # Minor #5-фикс: добавляем jitter ±50% к 429-backoff
                    # (аналогично network-error backoff) для предотвращения
                    # synchronized retries при одновременном спаде с нескольких
                    # клиентов.
                    jitter = random.uniform(-wait * 0.5, wait * 0.5)
                    actual = max(0.5, wait + jitter)
                    # Minor-фикс: логируем и спим только если будет retry.
                    # На последней попытке sleep впустую тратит до 120 с.
                    if attempt < self.max_retries:
                        logger.warning(
                            "Попытка %d/%d: 429 Rate Limited, ждём %.1f сек "
                            "(base=%d, jitter=%+.1f)",
                            attempt,
                            self.max_retries,
                            actual,
                            wait,
                            jitter,
                        )
                        try:
                            time.sleep(actual)
                        except Exception:
                            pass
                        last_exc = e
                        continue  # retry: переходим к следующей итерации
                    # Последняя попытка: не спать, выходим из цикла (raise LLMError)
                    last_exc = e
                elif 500 <= status < 600:
                    last_exc = e
                    logger.warning(
                        "Попытка %d/%d: HTTP %d от LLM API",
                        attempt,
                        self.max_retries,
                        status,
                    )
                elif status == 400:
                    # Major #2-фикс: НЕ любой HTTP 400 — контекстная длина.
                    # CRITICAL-фикс: классификация по телу ответа, чтобы избежать
                    # сплит-шторма при 400 "invalid_request_error" /
                    # "unsupported response_format". Проверяем только уточнённые
                    # маркеры: "context_length", "context_length_exceeded",
                    # "maximum context", "too long" (по контексту), "reduce the length".
                    resp_text = ""
                    try:
                        resp_text = e.response.text.lower()
                    except Exception:
                        pass
                    ctx_markers = (
                        "context_length",
                        "maximum context",
                        "too long",
                        "reduce the length",
                    )
                    if any(m in resp_text for m in ctx_markers):
                        raise LLMContextLengthError(
                            f"HTTP 400: контекстная длина превышена "
                            f"(max_tokens={self.max_tokens})"
                        ) from e
                    # Прочие 400 — общая ошибка, без сплита
                    body_text = ""
                    try:
                        body_text = e.response.text[:200]
                    except Exception:
                        body_text = "(тело недоступно)"
                    raise LLMError(
                        f"HTTP 400: {body_text}"
                    ) from e
                elif status in (401, 402, 403):
                    # Major #3-фикс: фатальные HTTP-ошибки авторизации/баланса.
                    # Retry бессмыслен: ситуация не исправится сама по себе.
                    # Выделяем в отдельный тип — вызывающая сторона знает,
                    # что items остаются pending, а не ok=False.
                    body_text = ""
                    try:
                        body_text = e.response.text[:200]
                    except Exception:
                        body_text = "(тело недоступно)"
                    raise LLMFatalError(
                        f"HTTP {status}: {body_text}"
                    ) from e
                else:
                    # Прочие 4xx — безнадёжно
                    body_text = ""
                    try:
                        body_text = e.response.text[:200]
                    except Exception:
                        body_text = "(тело недоступно)"
                    raise LLMError(f"HTTP {status}: {body_text}") from e

            except httpx.RequestError as e:
                # ConnectError, ReadError, RemoteProtocolError, UnsupportedProtocol…
                # Все транзиентные сетевые ошибки — ретраим
                last_exc = e
                logger.warning(
                    "Попытка %d/%d: сетевая ошибка при обращении к LLM API: %s",
                    attempt,
                    self.max_retries,
                    type(e).__name__,
                )

            except json.JSONDecodeError as e:
                last_exc = e
                # Minor #3-фикс: ровно 1 retry на JSONDecodeError.
                # Флаг _json_retry_done блокирует второй retry:
                #   - первая ошибка + ещё есть попытки → backoff 1 сек + ретраим 1 раз
                #   - уже был retry ИЛИ последняя попытка → break, отдаём last_exc
                if not _json_retry_done and attempt < self.max_retries:
                    _json_retry_done = True
                    logger.warning(
                        "Попытка %d/%d: невалидный JSON в ответе — повтор (1 из 2)",
                        attempt,
                        self.max_retries,
                    )
                    time.sleep(1)  # backoff перед retry (аналогично сетевым ошибкам)
                    continue
                # Второй JSONDecodeError или последняя попытка — выходим
                break

            except ValueError as e:
                # Minor 4-волны: _reject_json_constant бросил ValueError
                # (NaN/Infinity/-Infinity в JSON — не валидный RFC 8259).
                # Явная ветка ВЫШЕ catch-all фиксирует намерение: это
                # фатальная ошибка ответа, ретраить бессмысленно (модель
                # вернёт тот же литерал, токены уже списаны).
                # Ветка json.JSONDecodeError стоит раньше осознанно:
                # JSONDecodeError — подкласс ValueError, у него свой
                # 1-retry путь, семантика ретраев не меняется.
                raise LLMError(
                    f"Недопустимая JSON-константа в ответе: {e}"
                ) from e

            # ── LLMError и прочие ошибки — без ретрая, сразу наружу ──
            except LLMError:
                raise
            except Exception as e:
                # защита: любой неожиданный вылет — оборачиваем в LLMError
                raise LLMError(f"Неожиданная ошибка при обработке ответа API: {e}") from e

            # экспоненциальный backoff между ретраями (если ещё есть попытки)
            if attempt < self.max_retries:
                backoff = 2 ** (attempt - 1)
                # Minor #5-фикс: добавляем jitter ±50% для предотвращения
                # synchronized retries при одновременном спаде с нескольких
                # клиентов или при watchdog-перезапуске.
                jitter = random.uniform(-backoff * 0.5, backoff * 0.5)
                actual = max(0.5, backoff + jitter)
                logger.debug(
                    "Backoff %.1f сек (base=%d, jitter=%+.1f) перед следующей попыткой",
                    actual, backoff, jitter,
                )
                try:
                    time.sleep(actual)
                except Exception:
                    # Minor #1-фикс: защита от неожиданных исключений при sleep.
                    pass

        # все попытки исчерпаны
        # (ветка last_exc is None больше невозможна — max_retries <= 0
        # нормализуется в 1 попытку выше по коду; self.max_retries >= 1)
        exc_desc = (
            f"таймаут ({type(last_exc).__name__})"
            if isinstance(last_exc, httpx.TimeoutException)
            else type(last_exc).__name__
        )
        raise LLMError(
            f"LLM-вызов не удался после {self.max_retries} попыток: {exc_desc}"
        )

    def get_balance(self) -> BalanceInfo:
        """
        Проверяет баланс аккаунта через /user/balance.

        Возвращает
        -----------
        BalanceInfo (is_available=False при любой ошибке)
        """
        try:
            response = self._get("/user/balance")
            raw: Any = response.json()

            # Защита top-level
            if not isinstance(raw, dict):
                logger.warning(
                    "Некорректный формат ответа баланса: ожидался dict, "
                    "получен %s",
                    type(raw).__name__,
                )
                return BalanceInfo(is_available=False, currency="", total_balance=0.0)

            balance_infos: Any = raw.get("balance_infos")
            if not isinstance(balance_infos, list) or not balance_infos:
                # пустой balance_infos ≠ ошибка, но и не валидный баланс → is_available=False
                # чтобы pipeline не слал ложный алерт о «баланс 0.0»
                return BalanceInfo(is_available=False, currency="", total_balance=0.0)

            entry: Any = balance_infos[0]
            if not isinstance(entry, dict):
                logger.warning("balance_infos[0] не dict: %s", type(entry).__name__)
                return BalanceInfo(is_available=False, currency="", total_balance=0.0)

            currency = str(entry.get("currency", ""))
            raw_balance = entry.get("total_balance")

            # Minor-фикс: при отсутствующем или невалидном total_balance
            # возвращаем is_available=False — чтобы pipeline не слал ложный
            # алерт «баланс 0.0» (логика та же, что и для пустого balance_infos
            # выше по коду — пустой balance ≠ ошибка, но и не валидный баланс).
            if raw_balance is None:
                logger.warning("total_balance отсутствует в ответе баланса")
                return BalanceInfo(is_available=False, currency="", total_balance=0.0)

            try:
                total_balance = float(raw_balance)
            except (TypeError, ValueError):
                logger.warning(
                    "Некорректное значение total_balance: %r", raw_balance
                )
                return BalanceInfo(is_available=False, currency="", total_balance=0.0)

            # FIX-5-5: NaN/inf из ответа баланса (float("NaN") / float("inf"))
            # проходят без исключения, но is_available=True, total_balance=nan.
            # math.isfinite отсекает оба случая.
            if not math.isfinite(total_balance):
                logger.warning(
                    "total_balance невалиден (NaN/inf): %r", raw_balance
                )
                return BalanceInfo(is_available=False, currency="", total_balance=0.0)

            # Учитываем is_available из ответа API (если есть).
            # DeepSeek API иногда возвращает поле is_available — оно точно
            # отражает доступность средств (в отличие от пустого balance_infos
            # или нулевого total_balance, которые могут быть битым ответом).
            api_available = raw.get("is_available")
            if isinstance(api_available, bool):
                is_available = api_available
            else:
                # Если поля нет или оно не bool — принимаем как валидный ответ,
                # is_available=True (средства доступны по балансу).
                is_available = True

            return BalanceInfo(
                is_available=is_available,
                currency=currency,
                total_balance=total_balance,
            )

        except httpx.TimeoutException:
            logger.warning("Таймаут при проверке баланса LLM-аккаунта")
            return BalanceInfo(is_available=False, currency="", total_balance=0.0)

        except httpx.HTTPStatusError as e:
            logger.warning("Ошибка %d при проверке баланса: %s", e.response.status_code, e)
            return BalanceInfo(is_available=False, currency="", total_balance=0.0)

        except httpx.RequestError as e:
            logger.warning("Сетевая ошибка при проверке баланса: %s %s", type(e).__name__, e)
            return BalanceInfo(is_available=False, currency="", total_balance=0.0)

        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as e:
            logger.warning("Не удалось распарсить ответ баланса: %s", e)
            return BalanceInfo(is_available=False, currency="", total_balance=0.0)

        except Exception as e:
            # защита от непредвиденных вылетов
            logger.warning("Неожиданная ошибка get_balance: %s", e)
            return BalanceInfo(is_available=False, currency="", total_balance=0.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_usage_debug(self, data: Dict[str, Any]) -> None:
        """
        Логирует usage одной строкой на DEBUG. Безопасно при отсутствии полей.
        Также аккумулирует в self.usage_stats.
        """
        self.usage_stats.add_response(data)

        usage: Any = data.get("usage")
        if not isinstance(usage, dict):
            logger.debug(
                "LLM usage | model=%s | (usage unavailable)", self.model
            )
            return

        total = _safe_int(usage.get("total_tokens"), 0)
        completion = _safe_int(usage.get("completion_tokens"), 0)
        prompt = _safe_int(usage.get("prompt_tokens"), 0)

        cached_hit, cached_miss = _parse_cached_tokens(usage)

        logger.debug(
            "LLM usage | model=%s | prompt=%d (cached=%d, miss=%d) | completion=%d | total=%d",
            self.model, prompt, cached_hit, cached_miss, completion, total,
        )

    def _post(self, path: str, payload: Dict[str, Any]) -> httpx.Response:
        """POST с правильным заголовком авторизации."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.Client(timeout=self.timeout, trust_env=True)  # подхватывает HTTP_PROXY / HTTPS_PROXY из окружения (Docker)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self._http.post(f"{self.base_url}{path}", json=payload, headers=headers)
        response.raise_for_status()
        return response

    def _get(self, path: str) -> httpx.Response:
        """GET с правильным заголовком авторизации."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.Client(timeout=self.timeout, trust_env=True)  # подхватывает HTTP_PROXY / HTTPS_PROXY из окружения (Docker)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = self._http.get(f"{self.base_url}{path}", headers=headers)
        response.raise_for_status()
        return response