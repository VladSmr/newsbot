"""
DeepSeek API-клиент (OpenAI-compatible).
Все сетевые и парсинговые ошибки оборачиваются в LLMError или
корректный "пустой" BalanceInfo — вызывающая сторона не падает.
"""
import json
import logging
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger("newsbot.llm.client")


class LLMError(Exception):
    """Единая ошибка LLM-слоя: любая неудача при вызове API или парсинге ответа."""

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
    model : str      — имя модели, например deepseek-v4-pro
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
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        # один HTTP-клиент на весь жизненный цикл — переиспользуем соединения
        self._http = httpx.Client(timeout=self.timeout, trust_env=True)  # подхватывает HTTP_PROXY / HTTPS_PROXY из окружения (Docker)

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
            "response_format": {"type": "json_object"},
        }

        last_exc: Optional[Exception] = None

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
                # T2-10-фикс: JSONDecodeError — бессмысленно ретраить (модель
                # вернула битый JSON — в следующий раз та же модель даст тот же
                # ответ). Ретраим только 1 раз.
                if attempt < self.max_retries:
                    logger.warning(
                        "Попытка %d/%d: невалидный JSON в ответе (осталось %d) — повтор",
                        attempt, self.max_retries, self.max_retries - attempt,
                    )
                break  # выходим из цикла, отдаём last_exc

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