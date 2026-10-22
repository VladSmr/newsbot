"""
TelegramPoller — long-poll getUpdates для приёма callback_query и команд.

Работает параллельно с основным циклом в отдельном daemon thread.
on_digest передаётся как callable (вызывается ОСНОВНЫМ потоком через event).

Важно: одновременно два getUpdates-потребителя (poller и --check-telegram)
запускать НЕЛЬЗЯ — будет 409 Conflict.
"""
import json
import logging
import re
import threading
import time
from typing import Callable, List, Optional

import httpx

logger = logging.getLogger("newsbot.telegram_io")

# ── HTTP-статусы ──────────────────────────────────────────────────────────────
_HTTP_CONFLICT = 409


# ── Исключения ───────────────────────────────────────────────────────────────

class TgApiError(Exception):
    """
    Безопасное исключение для HTTP-ошибок Telegram API (N4).
    Не содержит URL с токеном в сообщении.
    Поле `http_status` хранит числовой код для детекта 409.
    """

    def __init__(self, http_status: Optional[int], cause: Optional[Exception]):
        self.http_status = http_status
        self.cause = cause
        if http_status is not None:
            super().__init__(f"HTTP {http_status}")
        else:
            super().__init__(type(cause).__name__ if cause else "unknown")

# ── Token sanitizer ─────────────────────────────────────────────────────────

def _TokenSanitizer(token: str):
    """Маскирует токен: показываем тип + последние 4 символа (N4)."""
    def sanitize(*args, **kwargs) -> str:
        if not token:
            return "(empty)"
        last4 = token[-4:] if len(token) >= 4 else token
        return f"...{last4} (len={len(token)})"
    return sanitize

def _safe_status_log(resp: httpx.Response) -> str:
    """Safe repr ответа без URL с токеном."""
    return f"<Response [{resp.status_code}]>"


# ── Telegram client ──────────────────────────────────────────────────────────

class _TgClient:
    """
    Minimal HTTP-клиент для Telegram Bot API.
    Ошибки логируют только тип (N4: без str(e) — содержит URL с токеном).
    """

    def __init__(self, token: str, timeout: float = 30.0):
        self.token = token
        self._base = f"https://api.telegram.org/bot{token}"
        self._http = httpx.Client(timeout=timeout, trust_env=True)  # подхватывает HTTP_PROXY / HTTPS_PROXY из окружения (Docker)

    def _call(self, method: str, payload: Optional[dict] = None) -> dict:
        """
        Вызывает Telegram Bot API.
        Raises TgApiError на всех HTTP-ошибках (безопасное сообщение, без URL/токена).
        Raises ValueError на ok=False.
        """
        url = f"{self._base}/{method}"
        try:
            resp = self._http.post(url, json=payload or {})
            resp.raise_for_status()  # HTTPStatusError на 4xx/5xx
        except httpx.HTTPStatusError as e:
            # N4: НЕ ре-рейзим сырое исключение (str содержит URL с токеном).
            # Сохраняем status_code для детекта 409 в _poll.
            raise TgApiError(
                http_status=e.response.status_code,
                cause=e,
            )
        except httpx.HTTPError as e:
            # прочие сетевые ошибки (Timeout, Connect…)
            raise TgApiError(http_status=None, cause=e)

        try:
            data = resp.json()
        except Exception:
            raise TgApiError(http_status=None, cause=None)

        if not data.get("ok"):
            logger.debug("TG %s → ok=false: %s", method, data.get("description", "?"))
            raise ValueError(f"Telegram error: {data.get('description', '?')}")

        return data.get("result", {})

    def getUpdates(
        self,
        offset: Optional[int] = None,
        timeout: int = 25,
        allowed_updates: Optional[List[str]] = None,
    ) -> List[dict]:
        payload = {"timeout": timeout}
        if offset is not None:
            payload["offset"] = offset
        if allowed_updates:
            payload["allowed_updates"] = json.dumps(allowed_updates)
        result = self._call("getUpdates", payload)
        return result if isinstance(result, list) else []

    def answerCallbackQuery(self, callback_query_id: str, text: str) -> None:
        self._call("answerCallbackQuery", {
            "callback_query_id": callback_query_id,
            "text": text,
            "show_alert": False,
        })

    def sendMessage(self, chat_id: str, text: str) -> None:
        self._call("sendMessage", {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        })

    def editMessageReplyMarkup(
        self,
        chat_id: str,
        message_id: int,
        reply_markup: Optional[dict] = None,
    ) -> None:
        payload = {
            "chat_id": chat_id,
            "message_id": message_id,
            "reply_markup": (
                reply_markup if reply_markup is not None
                else {"inline_keyboard": []}
            ),
        }
        try:
            self._call("editMessageReplyMarkup", payload)
        except Exception:
            pass  # best-effort

    def close(self) -> None:
        self._http.close()


# ── Callbacks data parser ─────────────────────────────────────────────────────

_CALLBACK_PATTERN = re.compile(r"^fb:(\d+):([ud])$")


def parse_feedback_callback(data: str):
    """
    Парсит callback_data кнопки 👍/👎.
    Returns (target_id: int, verdict: 'like'|'dislike') или None.
    """
    m = _CALLBACK_PATTERN.match(data)
    if not m:
        return None
    return (int(m.group(1)), "like" if m.group(2) == "u" else "dislike")


# ── Команды ───────────────────────────────────────────────────────────────────

class _Commands:
    KNOWN = {"/start", "/help", "/digest", "/stop", "/continue"}

    @staticmethod
    def is_owner_command(msg: dict, owner_chat_id: str) -> bool:
        """True если сообщение — команда от владельца (C2: безопасный парсинг)."""
        chat = msg.get("chat", {})
        if str(chat.get("id", "")) != str(owner_chat_id):
            return False
        text = msg.get("text", "") or ""
        parts = text.strip().split()
        # C2: защита от IndexError (фото/стикер/service-message без текста)
        return bool(parts) and parts[0] in _Commands.KNOWN

    @staticmethod
    def text(msg: dict) -> str:
        """Текст сообщения (без leading / command)."""
        return (msg.get("text", "") or "").strip()


# ── TelegramPoller ────────────────────────────────────────────────────────────

class TelegramPoller:
    """
    Long-poll поллер Telegram.

    Обрабатывает:
      — callback_query с data='fb:{id}:{u|d}' (только от владельца, M2)
      — команды /stop, /continue, /digest, /start, /help от владельца
      — /stop и /continue пишут флаг 'paused' в meta (shared с pipeline)

    Не падает при сетевых ошибках, таймаутах, 409 Conflict.

    Thread-safety: poller создаёт СОБСТВЕННОЕ соединение Storage в своём
    потоке (C1: отдельный экземпляр, а не обёртка над main-connection).
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        db_path: str,
        on_digest: Callable[[], None],
        logger: logging.Logger = None,
        poll_timeout: int = 25,
        feedback_enabled: bool = True,
    ):
        if not logger:
            logger = logging.getLogger("newsbot.telegram_io")
        self._log = logger
        # MINOR-фикс: HTTP-таймаут клиента должен быть БОЛЬШЕ long-poll таймаута
        # getUpdates (poll_timeout), иначе read-timeout клиента срабатывает
        # одновременно с ответом Telegram → ложный TgApiError(timeout) → backoff
        # и повторная обработка тех же updates (двойной /stop, двойной фидбэк).
        # poll_timeout клампится в config.py (≤30) → HTTP-таймаут ≤40 с.
        self._client = _TgClient(token, timeout=float(poll_timeout) + 10.0)
        self._chat_id = str(chat_id)
        self._db_path = db_path
        # on_digest — callable для установки флага (НЕ выполнения в poller thread)
        self._on_digest = on_digest
        self._poll_timeout = poll_timeout
        self._feedback_enabled = feedback_enabled

        # Storage создаётся ЛЕНИВО в poller thread (C1)
        self._storage = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # m2: backoff как атрибут, чтобы _poll мог вызвать _sleep_backoff.
        # Инициализируется ОДИН раз в __init__ (НЕ в _run_loop) — это позволяет
        # тестам устанавливать начальное значение перед стартом цикла.
        self._backoff: float = 1.0

    # ── public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            self._log.warning("Poller already started")
            return
        self._stop_event.clear()
        t = threading.Thread(target=self._run_loop, name="TelegramPoller", daemon=True)
        t.start()
        self._thread = t
        self._log.info("TelegramPoller started (daemon thread)")

    def stop(self, max_join: Optional[float] = None) -> None:
        """
        Останавливает poller.

        Параметры
        ---------
        max_join : float, optional
            Верхняя граница join (сек), обычно остаток единого shutdown-бюджета
            из __main__. Итоговый таймаут = min(poll_timeout + 5, max_join) —
            так повторный join не выходит за жёсткий дедлайн (P1-фикс:
            раньше join(poll_timeout+5) выполнялся ДО расчёта remaining и
            бюджет 40 с не был жёстким).
        """
        self._log.info("Stopping TelegramPoller...")
        self._stop_event.set()
        if self._thread is not None:
            # Minor-фикс: join с учётом poll_timeout. Long-poll может висеть
            # до poll_timeout секунд (~25 с); удлиняем timeout соответственно.
            # НЕ обнуляем _thread если поток ещё alive — это защита от
            # race с _run_loop: close() может вызваться когда _run_loop ещё
            # работает и _thread ещё жив, но stop() уже отпустил GIL.
            # Вместо этого ждём реального завершения потока.
            join_timeout = self._poll_timeout + 5
            if max_join is not None:
                join_timeout = max(0.0, min(join_timeout, float(max_join)))
            self._thread.join(timeout=join_timeout)
            # Только если поток ДЕЙСТВИТЕЛЬНО завершился — обнуляем.
            # Иначе оставляем reference alive (не создаём второй поллер).
            if not self._thread.is_alive():
                self._thread = None
                self._log.info("TelegramPoller stopped")
            else:
                self._log.warning(
                    "TelegramPoller thread still alive after %.1f s timeout "
                    "(poll_timeout=%d). Leaving thread reference (no second poller).",
                    join_timeout, self._poll_timeout,
                )

    # ── poller thread ────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        """
        Основной long-poll цикл (C1: Storage создаётся здесь, в poller thread).
        m2: Storage.close() вызывается ВНУТРИ poller thread (finally).

        backoff-логика:
          — успех (_tick → True): сброс в 1.0
          — ошибка (_tick → False): backoff *= 2, sleep на текущем значении
          — пустой тик / 409 (_tick → None): backoff не трогается
          — 409 внутри _poll: _poll уже спит 60с, _tick возвращает None,
            _run_loop НЕ спит и НЕ удваивает backoff
        """
        # C1: создаём Storage в poller thread — своё соединение, не main.
        try:
            from newsbot.storage import Storage
            self._storage = Storage(self._db_path)
            self._log.debug("Poller Storage created in thread (db=%s)", self._db_path)
        except Exception as e:
            self._log.error("Failed to create Storage in poller thread: %s", e)
            return  # не можем работать без storage

        # m2: Storage.close() в finally — гарантированно в poller thread.
        try:
            # m2-backoff: инициализируется в __init__ (self._backoff = 1.0).
            # При успехе сбрасывается здесь: self._backoff = 1.0.
            while not self._stop_event.is_set():
                result = self._tick(self._backoff)
                if result is True:
                    # успех — сброс backoff
                    self._backoff = 1.0
                elif result is False:
                    # ошибка — растём, sleep на текущем значении
                    self._sleep_backoff(self._backoff)
                    self._backoff = min(self._backoff * 2, 60.0)
                # result is None (empty updates): backoff не трогаем
        finally:
            # m2: закрываем наше соединение в poller thread (не в main)
            if self._storage is not None:
                try:
                    self._storage.close()
                    self._log.debug("Poller Storage closed in poller thread")
                except Exception:
                    pass
                self._storage = None

    def _tick(self, backoff: float) -> Optional[bool]:
        """
        Один виток: poll → process updates. Каждый шаг защищён.

        Returns:
          True  — updates обработаны (успех)
          False — ошибка (network/HTTP, кроме 409).
                   _run_loop делает sleep + backoff*=2.
          None  — пустой тик (empty updates) ИЛИ 409 Conflict
                   (сбой другого потребителя). _run_loop НЕ спит и НЕ растёт.
        """
        # C1: _load_offset может бросить (если storage ещё не создан)
        try:
            offset = self._load_offset()
        except Exception as e:
            self._log.warning("_load_offset failed: %s", e)
            offset = None

        try:
            updates = self._poll(offset)
        except TgApiError as e:
            # N1: 409 определяется по http_status через TgApiError.
            # _poll уже сделал sleep 60с при 409 — нейтральный return.
            if e.http_status == _HTTP_CONFLICT:
                self._log.error(
                    "409 Conflict: another getUpdates consumer active. Pause 60 s."
                )
                # return None: _run_loop НЕ спит и НЕ удваивает backoff.
                return None
            # прочие HTTP-ошибки: backoff растёт в _run_loop
            self._log.warning(
                "Network error during poll (HTTP %s), backoff %.1f s",
                e.http_status, backoff,
            )
            return False
        except Exception as e:
            self._log.error("Poll error: %s", e)
            return False

        if not updates:
            return None  # пустой тик, backoff не трогаем

        for upd in updates:
            # C2: каждый update защищён — ошибка не убивает поток
            try:
                self._process_update(upd)
            except Exception as e:
                self._log.warning(
                    "Error processing update %s: %s",
                    upd.get("update_id", "?"), e,
                )

        # сохраняем offset
        max_id = max(u.get("update_id", 0) for u in updates)
        self._save_offset(max_id + 1)
        return True  # успех: updates обработаны

    def _poll(self, offset: Optional[int]) -> List[dict]:
        """
        Один long-poll запрос.
        TgApiError НЕ ловится — всплывает в _tick, где есть обработка ошибок
        и backoff. Это позволяет:
          (a) ветке except TgApiError в _tick работать;
          (b) backoff накапливаться при серии ошибок.
        Исключение: 409 — здесь же логируем и спим 60с, затем re-raise.
        _tick ловит re-raise и возвращает None (нейтрально, без backoff).
        """
        try:
            return self._client.getUpdates(
                offset=offset,
                timeout=self._poll_timeout,
                allowed_updates=["callback_query", "message"],
            )
        except TgApiError as e:
            if e.http_status == _HTTP_CONFLICT:
                self._log.warning(
                    "getUpdates: 409 Conflict — another consumer active. Pause 60 s."
                )
                self._sleep_backoff(60.0)
            # ошибка всплывает — _tick обработает общий случай
            raise

    # ── update dispatch ───────────────────────────────────────────────────────

    def _process_update(self, upd: dict) -> None:
        """Диспетчеризирует update: callback_query или message."""
        if "callback_query" in upd:
            self._handle_callback_query(upd["callback_query"])
        elif "message" in upd:
            self._handle_message(upd["message"])

    # ── callback_query (M2: проверка владельца) ──────────────────────────────

    def _handle_callback_query(self, cb: dict) -> None:
        """
        Обрабатывает нажатие кнопки 👍/👎.

        Формат data: fb:{target_id}:{u|d}
        C2: обёрнуто в try/except.
        M2: проверяем chat_id (чужой чат — игнор).
        """
        query_id = cb.get("id", "")
        data = cb.get("data", "")
        message = cb.get("message", {})

        # M2: проверка владельца
        chat_id = str(message.get("chat", {}).get("id", ""))
        if chat_id != self._chat_id:
            self._log.debug("Callback from wrong chat %s (owner=%s) — ignored",
                           chat_id, self._chat_id)
            return

        parsed = parse_feedback_callback(data)
        if parsed is None:
            self._log.debug("Invalid callback data: %r", data)
            return

        target_id, verdict = parsed
        message_id = message.get("message_id")

        # C2: обёртка try/except для record_feedback
        # n3: гейтится по feedback_enabled (параметр из __main__), независимо от
        # feedback.enabled в конфиге — поллер стартует без feedback.enabled.
        recorded = False
        if self._feedback_enabled:
            try:
                recorded = self._storage.record_feedback(
                    target_id=target_id,
                    verdict=verdict,
                    chat_id=chat_id,
                    message_id=message_id,
                )
            except Exception as e:
                self._log.warning("record_feedback failed for target=%d: %s",
                                 target_id, e)

        popup_text = "👍 Учту" if verdict == "like" else "👎 Понял, такое больше не присылать"

        try:
            self._client.answerCallbackQuery(query_id, text=popup_text)
        except Exception as e:
            self._log.warning("answerCallbackQuery failed for query=%s: %s",
                             query_id[:20], type(e).__name__)

        # n5: editMessageReplyMarkup уже best-effort внутри _call — внешний try/else не нужен
        if recorded and message_id:
            # Строим новую клавиатуру: сохраняем URL-кнопки, удаляем только 👍/👎
            # для нажатого target_id. Чужие fb:-кнопки (другие target_id) сохраняются.
            raw_markup = message.get("reply_markup")
            if isinstance(raw_markup, dict) and "inline_keyboard" in raw_markup:
                old_kb = raw_markup["inline_keyboard"]
                if isinstance(old_kb, list):
                    new_kb = []
                    for row in old_kb:
                        if not isinstance(row, list):
                            continue
                        new_row = [
                            btn for btn in row
                            if "callback_data" not in btn
                            or not (
                                btn.get("callback_data", "").startswith(f"fb:{target_id}:")
                            )
                        ]
                        if new_row:
                            new_kb.append(new_row)

                    if new_kb:
                        self._client.editMessageReplyMarkup(
                            chat_id=chat_id,
                            message_id=message_id,
                            reply_markup={"inline_keyboard": new_kb},
                        )
                    else:
                        # пустая клавиатура (URL-кнопок не было) — очищаем
                        self._client.editMessageReplyMarkup(
                            chat_id=chat_id,
                            message_id=message_id,
                            reply_markup={"inline_keyboard": []},
                        )

        self._log.debug("feedback: target=%d verdict=%s recorded=%s",
                       target_id, verdict, recorded)

    # ── message commands ─────────────────────────────────────────────────────

    def _handle_message(self, msg: dict) -> None:
        """Обрабатывает команды от владельца."""
        if not _Commands.is_owner_command(msg, self._chat_id):
            return

        cmd, *_ = _Commands.text(msg).split(maxsplit=1)
        chat_id = str(msg.get("chat", {}).get("id", ""))

        if cmd == "/stop":
            self._log.info("Command /stop — pausing processing")
            try:
                if self._storage is not None:
                    self._storage.set_meta("paused", "1")
                self._client.sendMessage(
                    chat_id,
                    "⏸ Приостановлено: новые новости собираются, но не "
                    "обрабатываются и не присылаются. /continue — возобновить."
                )
            except Exception as e:
                self._log.warning("/stop failed: %s", type(e).__name__)

        elif cmd == "/continue":
            self._log.info("Command /continue — resuming processing")
            try:
                if self._storage is not None:
                    self._storage.set_meta("paused", "0")
                self._client.sendMessage(chat_id, "▶️ Возобновлено.")
            except Exception as e:
                self._log.warning("/continue failed: %s", type(e).__name__)

        elif cmd == "/digest":
            # ── проверка паузы: /digest не выполняется при паузе ──
            is_paused = False
            try:
                if self._storage is not None:
                    val = self._storage.get_meta("paused")
                    is_paused = val == "1"
            except Exception as e:
                self._log.warning("_check_paused failed: %s", e)

            if is_paused:
                self._log.debug("/digest ignored — bot is paused")
                try:
                    self._client.sendMessage(
                        chat_id,
                        "⏸ Бот на паузе, /continue чтобы возобновить."
                    )
                except Exception:
                    pass
                return  # M1: НЕ дёргаем _on_digest

            self._log.info("Command /digest from owner — signaling main thread")
            try:
                self._on_digest()  # M1: только сигнал, НЕ run_digest в poller thread
                self._client.sendMessage(
                    chat_id,
                    "📰 Команда принята, дайджест будет выполнен..."
                )
            except Exception as e:
                self._log.error("/digest signal failed: %s", e)
                try:
                    self._client.sendMessage(chat_id, "❌ Ошибка при запуске дайджеста.")
                except Exception:
                    pass

        elif cmd in ("/start", "/help"):
            greeting = (
                "👋 <b>newsbot</b> — персональный новостной агрегатор.\n\n"
                "📬 Новости приходят автоматически по расписанию.\n"
                "👍👎 Голосуйте за новости — это улучшает подборку.\n\n"
                "📋 <b>Команды:</b>\n"
                "/stop     — приостановить обработку и доставку\n"
                "/continue — возобновить после /stop\n"
                "/digest   — запустить дайджест отложенных новостей\n"
                "/help     — эта справка"
            )
            try:
                self._client.sendMessage(chat_id, greeting)
            except Exception as e:
                self._log.warning("sendMessage (start/help) failed: %s", type(e).__name__)

        else:
            self._log.debug("Unknown command ignored: %s", cmd)

    # ── offset persistence ────────────────────────────────────────────────────

    META_OFFSET_KEY = "tg_updates_offset"

    def _load_offset(self) -> Optional[int]:
        if self._storage is None:
            return None
        try:
            val = self._storage.get_meta(self.META_OFFSET_KEY)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            self._log.warning("_load_offset failed: %s", e)
        return None

    def _save_offset(self, offset: int) -> None:
        if self._storage is None:
            return
        try:
            self._storage.set_meta(self.META_OFFSET_KEY, str(offset))
        except Exception as e:
            self._log.warning("_save_offset failed: %s", e)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _sleep_backoff(self, seconds: float) -> None:
        """Спит seconds с проверкой stop_event."""
        step = 1.0
        slept = 0.0
        while slept < seconds and not self._stop_event.is_set():
            time.sleep(min(step, seconds - slept))
            slept += step

    def close(self, join_timeout: Optional[float] = None) -> None:
        """
        Закрывает HTTP-клиент poller.
        m2: НЕ закрывает Storage — это делает _run_loop в finally (в poller thread).
        Minor-фикс: выставляем _stop_event ДО join — защита от сценария,
        когда close() вызвана БЕЗ предварительного stop().
        Без set(): если поток в long-poll (25 с), close() без stop() ждёт
        все 30 с — лишнее ожидание. С set() поток проснётся сразу (через
        _sleep_backoff с is_set()-проверкой).

        Параметры
        ----------
        join_timeout : float, optional
            Явный таймаут join. Нужен для единого дедлайна shutdown: если
            stop() уже ждал поток, close() не должен повторно ждать полные
            poll_timeout+5. None (по умолчанию) — прежнее поведение.
        """
        # Minor-фикс: выставить флаг ДО join — снимает лишнее ожидание
        # и защищает от close() без предварительного stop().
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._log.debug("close: waiting for poller thread to finish...")
            timeout = (
                self._poll_timeout + 5
                if join_timeout is None
                else max(0.0, float(join_timeout))
            )
            self._thread.join(timeout=timeout)
        self._client.close()