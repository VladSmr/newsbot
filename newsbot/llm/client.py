"""
DeepSeek API-клиент (OpenAI-compatible).
Все сетевые и парсинговые ошибки оборачиваются в LLMError или
корректный "пустой" BalanceInfo — вызывающая сторона не падает.
"""
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx

logger = logging.getLogger("newsbot.llm.client")


# ─────────────────────────────────────────────────────────────────
# Usage accumulator
# ─────────────────────────────────────────────────────────────────


def _safe_int(value: Any, default: int = 0) -> int:
    """Безопасное преобразование в int. При ошибке — default."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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

    def set_prices(self, input_price: float, output_price: float) -> None:
        """Устанавливает цены модели (USD за 1M токенов)."""
        self._input_price = input_price
        self._output_price = output_price

    @property
    def total_cost_usd(self) -> float:
        """
        Вычисляет общую стоимость всех вызовов в USD.
        Формула: (prompt_tokens * input + completion_tokens * output) / 1_000_000
        Кэшированные токены не вычитаются (cached токены стоят дешевле,
        но модель считает их по полной в prompt_tokens).
        Для учёта кэшированных токенов используйте реальные цены DeepSeek
        с дисконтом на cached (input * 0.1).
        """
        cost = (
            self.prompt_tokens * self._input_price
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
        max_tokens: int = 4096,
        input_price: float = 0.0,
        output_price: float = 0.0,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        # max_tokens: лимит на completion. Батч (≤10 items, ~100-200 токенов/item)
        # требует больше, чем одиночная оценка (~150-300 токенов).
        # Major #2-фикс: дефолт 4096 (ранее 2048).
        # При batch_size=10: ≈400-600 токенов/item × 10 = 4000-6000 токенов ответ,
        # модель обычно укладывается в 4000, запас до 4096 покрывает типичный
        # батч 10 items + погрешность модели. Конфиг позволяет переопределить.
        self.max_tokens = max_tokens
        # один HTTP-клиент на весь жизненный цикл — переиспользуем соединения
        self._http = httpx.Client(timeout=self.timeout, trust_env=True)  # подхватывает HTTP_PROXY / HTTPS_PROXY из окружения (Docker)
        self.usage_stats = UsageStats(model=model)
        # Устанавливаем цены для расчёта стоимости
        self.usage_stats.set_prices(input_price, output_price)

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

        for attempt in range(1, self.max_retries + 1):
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

                # Major #2-фикс: проверяем finish_reason на усечение.
                # При length → JSON может быть обрезан → json.loads упадёт.
                # Поднимаем LLMError, чтобы вызывающий получил fallback per-item
                # (а не парсил битый JSON молча).
                finish_reason: Any = first_choice.get("finish_reason")
                if finish_reason == "length":
                    # usage уже аккумулирован выше (сразу после проверки dict) —
                    # токены учтены даже при length-truncation.
                    logger.warning(
                        "LLM ответ усечён (finish_reason=length, max_tokens=%d) — "
                        "fallback для батча: шлём непрофильтрованными",
                        self.max_tokens,
                    )
                    raise LLMTruncationError(
                        f"LLM-ответ усечён (finish_reason=length, max_tokens={self.max_tokens}). "
                        f"JSON может быть неполным — fallback (ok=False для всех items батча)."
                    )

                # ── usage-логирование уже выполнено выше (одно на response) ──

                result: Any = json.loads(content)
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
                            wait = int(retry_after)
                        except ValueError:
                            wait = 2 ** attempt
                    else:
                        wait = 2 ** attempt
                    # B8-фикс: cap wait на 120 сек (как в telegram-слое)
                    wait = min(wait, 120)
                    logger.warning(
                        "Попытка %d/%d: 429 Rate Limited, ждём %d сек",
                        attempt,
                        self.max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    last_exc = e
                    continue
                elif 500 <= status < 600:
                    last_exc = e
                    logger.warning(
                        "Попытка %d/%d: HTTP %d от LLM API",
                        attempt,
                        self.max_retries,
                        status,
                    )
                else:
                    # 4xx (кроме 429) — безнадёжно
                    raise LLMError(f"HTTP {status}: {e.response.text[:200]}")

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

            # ── LLMError и прочие ошибки — без ретрая, сразу наружу ──
            except LLMError:
                raise
            except Exception as e:
                # защита: любой неожиданный вылет — оборачиваем в LLMError
                raise LLMError(f"Неожиданная ошибка при обработке ответа API: {e}") from e

            # экспоненциальный backoff между ретраями (если ещё есть попытки)
            if attempt < self.max_retries:
                backoff = 2 ** (attempt - 1)
                logger.debug("Backoff %d сек перед следующей попыткой", backoff)
                time.sleep(backoff)

        # все попытки исчерпаны
        if last_exc is None:
            # max_retries == 0 — цикл не прошёл ни разу
            raise LLMError(
                f"LLM-вызов не выполнен: max_retries={self.max_retries} (цикл пропущен)"
            )
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
            if raw_balance is None:
                total_balance = 0.0
            else:
                try:
                    total_balance = float(raw_balance)
                except (TypeError, ValueError):
                    logger.warning(
                        "Некорректное значение total_balance: %r", raw_balance
                    )
                    total_balance = 0.0

            return BalanceInfo(
                is_available=True,
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