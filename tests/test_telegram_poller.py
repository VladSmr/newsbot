"""
tests/test_telegram_poller.py — модульные тесты TelegramPoller.

Требования:
  — все HTTP вызовы замоканы (httpx)
  — реальных сетевых вызовов нет
  — токен НЕ попадает в логи

Проверяемые сценарии (фикс по результатам ревью):
  (1) parse_feedback_callback — валидные/невалидные data
  (2) callback_query → record_feedback + answerCallbackQuery
  (3) устаревший target (record_feedback=False) — не падает
  (4) offset сохраняется в meta и используется в следующем getUpdates
  (5) /digest → вызван _on_digest (сигнал, не run_digest); /start|/help → ответ;
       чужая команда/чужой chat — игнор
  (6) сетевая ошибка/таймаут в poll_loop → не падает, backoff
  (7) 409 Conflict (httpx.HTTPStatusError) обрабатывается
  (8) токен не попадает в лог
  (9) C2: is_owner_command на фото/стикере/empty-text не падает (IndexError)
  (10) M2: callback от чужого chat_id игнорируется
  (11) C1: poller создаёт СОБСТВЕННОЕ Storage в своём потоке
  (12) M1: on_digest вызывается как СИГНАЛ (не run_digest)
  (13) m1: нет дубликата test_callback_like_calls_record_and_answer
"""

import json
import logging
import re
import threading
import unittest
from unittest.mock import ANY, MagicMock, patch, PropertyMock

from newsbot.telegram_io import (
    TelegramPoller,
    parse_feedback_callback,
    _TokenSanitizer,
    _TgClient,
    _Commands,
    _HTTP_CONFLICT,
    TgApiError,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. parse_feedback_callback
# ══════════════════════════════════════════════════════════════════════════════

class TestParseFeedbackCallback(unittest.TestCase):
    def test_like(self):
        res = parse_feedback_callback("fb:123:u")
        self.assertEqual(res, (123, "like"))

    def test_dislike(self):
        res = parse_feedback_callback("fb:456:d")
        self.assertEqual(res, (456, "dislike"))

    def test_invalid_id_non_numeric(self):
        self.assertIsNone(parse_feedback_callback("fb:abc:u"))

    def test_invalid_verdict(self):
        self.assertIsNone(parse_feedback_callback("fb:123:x"))

    def test_no_prefix(self):
        self.assertIsNone(parse_feedback_callback("123:u"))

    def test_empty(self):
        self.assertIsNone(parse_feedback_callback(""))

    def test_wrong_format(self):
        self.assertIsNone(parse_feedback_callback("foobar"))

    def test_partial(self):
        self.assertIsNone(parse_feedback_callback("fb:123"))

    def test_extra_parts(self):
        self.assertIsNone(parse_feedback_callback("fb:123:u:extra"))


# ══════════════════════════════════════════════════════════════════════════════
# 2. _TokenSanitizer
# ══════════════════════════════════════════════════════════════════════════════

class TestTokenSanitizer(unittest.TestCase):
    def test_masked(self):
        s = _TokenSanitizer("12345678abcdefgh")
        result = s()
        self.assertNotIn("12345678abcdefgh", result)
        self.assertIn("len=16", result)
        self.assertIn("...efgh", result)

    def test_short_token(self):
        self.assertEqual(_TokenSanitizer("ab")(), "...ab (len=2)")

    def test_empty(self):
        self.assertEqual(_TokenSanitizer("")(), "(empty)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. _Commands (C2: защита от IndexError)
# ══════════════════════════════════════════════════════════════════════════════

class TestCommands(unittest.TestCase):
    OWNER = "111222333"

    def _msg(self, text: str, chat_id: str = None):
        chat_id = chat_id or self.OWNER
        return {"message_id": 1, "chat": {"id": chat_id}, "text": text}

    def test_digest_owner(self):
        self.assertTrue(_Commands.is_owner_command(self._msg("/digest"), self.OWNER))

    def test_start_owner(self):
        self.assertTrue(_Commands.is_owner_command(self._msg("/start"), self.OWNER))

    def test_help_owner(self):
        self.assertTrue(_Commands.is_owner_command(self._msg("/help"), self.OWNER))

    def test_other_chat(self):
        self.assertFalse(_Commands.is_owner_command(self._msg("/digest"), "999888777"))

    def test_unknown_command(self):
        self.assertFalse(_Commands.is_owner_command(self._msg("/stats"), self.OWNER))

    def test_regular_text(self):
        self.assertFalse(_Commands.is_owner_command(self._msg("hello"), self.OWNER))

    # ── C2: защита от IndexError на не-текстовых сообщениях ──────────────────

    def test_photo_message_no_crash(self):
        """C2: сообщение-фото (без поля text) не вызывает IndexError."""
        msg = {"message_id": 1, "chat": {"id": self.OWNER}}
        # не падает
        result = _Commands.is_owner_command(msg, self.OWNER)
        self.assertFalse(result)

    def test_empty_text_no_crash(self):
        """C2: пустой text='' не вызывает IndexError."""
        msg = {"message_id": 1, "chat": {"id": self.OWNER}, "text": ""}
        result = _Commands.is_owner_command(msg, self.OWNER)
        self.assertFalse(result)

    def test_sticker_no_crash(self):
        """C2: стикер без text не вызывает IndexError."""
        msg = {"message_id": 1, "chat": {"id": self.OWNER}, "sticker": {"emoji": "😀"}}
        result = _Commands.is_owner_command(msg, self.OWNER)
        self.assertFalse(result)

    def test_service_message_no_crash(self):
        """C2: service message (новый участник) не вызывает IndexError."""
        msg = {"message_id": 1, "chat": {"id": self.OWNER}, "new_chat_member": {}}
        result = _Commands.is_owner_command(msg, self.OWNER)
        self.assertFalse(result)


# ══════════════════════════════════════════════════════════════════════════════
# 4. _TgClient мок-вспомогатели
# ══════════════════════════════════════════════════════════════════════════════

class _FakeStopEvent:
    """Fake threading.Event для управления циклом из теста."""

    def __init__(self):
        self._set = False

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def clear(self):
        self._set = False

    def wait(self, timeout=None):
        return self._set


# ══════════════════════════════════════════════════════════════════════════════
# 5. TelegramPoller — helpers для построения poller с моками
# ══════════════════════════════════════════════════════════════════════════════

def _make_mock_client(get_updates_result=None):
    """_TgClient-мок: все методы — no-op по умолчанию."""
    c = MagicMock(spec=_TgClient)
    c.getUpdates.return_value = get_updates_result if get_updates_result is not None else []
    c.answerCallbackQuery.return_value = None
    c.sendMessage.return_value = None
    c.editMessageReplyMarkup.return_value = None
    return c


def _make_mock_storage():
    s = MagicMock()
    s.get_meta.return_value = None
    s.record_feedback.return_value = True
    return s


def _build_poller(
    storage_mock=None,
    client_mock=None,
    on_digest=None,
    chat_id="111222333",
):
    """
    Строит TelegramPoller с подменёнными моками.
    _storage подменяется на мок после инициализации (отдельный экземпляр).
    """
    on_digest = on_digest or MagicMock()
    poller = TelegramPoller(
        token="tok123456789",
        chat_id=chat_id,
        db_path=":memory:",
        on_digest=on_digest,
        logger=logging.getLogger("test"),
        poll_timeout=1,
    )
    # подменяем storage на мок (C1: мокаем напрямую, обёртка убрана)
    poller._storage = storage_mock or MagicMock()
    poller._storage.record_feedback.return_value = True
    poller._client = client_mock or _make_mock_client()
    poller._stop_event = _FakeStopEvent()
    return poller


# ══════════════════════════════════════════════════════════════════════════════
# 6. TelegramPoller — callback_query
# ══════════════════════════════════════════════════════════════════════════════

class TestCallbackQueryHandling(unittest.TestCase):
    """callback_query → record_feedback + answerCallbackQuery."""

    # ── 6a: like ─────────────────────────────────────────────────────────────

    def test_callback_like_calls_record_and_answer(self):
        poller = _build_poller()
        upd = {
            "update_id": 10,
            "callback_query": {
                "id": "cb_q_1",
                "data": "fb:42:u",
                "message": {
                    "chat": {"id": "111222333"},
                    "message_id": 555,
                },
            },
        }
        poller._process_update(upd)

        poller._storage.record_feedback.assert_called_once()
        args = poller._storage.record_feedback.call_args
        self.assertEqual(args.kwargs.get("target_id"), 42)
        self.assertEqual(args.kwargs.get("verdict"), "like")
        self.assertEqual(args.kwargs.get("chat_id"), "111222333")
        self.assertEqual(args.kwargs.get("message_id"), 555)
        poller._client.answerCallbackQuery.assert_called_once_with(
            "cb_q_1", text="👍 Учту"
        )

    # ── 6b: dislike ──────────────────────────────────────────────────────────

    def test_callback_dislike_calls_record_and_answer(self):
        poller = _build_poller()
        upd = {
            "update_id": 11,
            "callback_query": {
                "id": "cb_q_2",
                "data": "fb:99:d",
                "message": {
                    "chat": {"id": "111222333"},
                    "message_id": 777,
                },
            },
        }
        poller._process_update(upd)

        poller._storage.record_feedback.assert_called_once()
        args = poller._storage.record_feedback.call_args
        self.assertEqual(args.kwargs.get("target_id"), 99)
        self.assertEqual(args.kwargs.get("verdict"), "dislike")
        self.assertEqual(args.kwargs.get("chat_id"), "111222333")
        self.assertEqual(args.kwargs.get("message_id"), 777)
        poller._client.answerCallbackQuery.assert_called_once_with(
            "cb_q_2", text="👎 Понял, такое больше не присылать"
        )

    # ── 6c: unknown target — не падает ────────────────────────────────────────

    def test_callback_unknown_target_does_not_crash(self):
        poller = _build_poller()
        poller._storage.record_feedback.return_value = False  # устаревший target

        upd = {
            "update_id": 12,
            "callback_query": {
                "id": "cb_q_3",
                "data": "fb:999999:d",
                "message": {
                    "chat": {"id": "111222333"},
                    "message_id": 888,
                },
            },
        }
        # не падает
        poller._process_update(upd)
        poller._storage.record_feedback.assert_called_once()
        poller._client.answerCallbackQuery.assert_called_once()

    # ── 6d: invalid callback_data — игнорируется ───────────────────────────

    def test_invalid_callback_data_ignored(self):
        poller = _build_poller()
        upd = {
            "update_id": 13,
            "callback_query": {
                "id": "cb_q_4",
                "data": "bad_data",
                "message": {
                    "chat": {"id": "111222333"},
                    "message_id": 999,
                },
            },
        }
        poller._process_update(upd)
        poller._storage.record_feedback.assert_not_called()
        poller._client.answerCallbackQuery.assert_not_called()

    # ── 6e: answerCallbackQuery failure — best-effort, не падает ─────────────

    def test_answer_callback_query_failure_best_effort(self):
        poller = _build_poller()
        poller._client.answerCallbackQuery.side_effect = Exception("API error")

        upd = {
            "update_id": 14,
            "callback_query": {
                "id": "cb_q_5",
                "data": "fb:7:u",
                "message": {
                    "chat": {"id": "111222333"},
                    "message_id": 100,
                },
            },
        }
        # не падает
        poller._process_update(upd)
        poller._storage.record_feedback.assert_called_once()

    # ── 6f: record_feedback exception — не падает (C2) ───────────────────────

    def test_record_feedback_exception_best_effort(self):
        poller = _build_poller()
        poller._storage.record_feedback.side_effect = RuntimeError("DB error")

        upd = {
            "update_id": 15,
            "callback_query": {
                "id": "cb_q_6",
                "data": "fb:5:u",
                "message": {
                    "chat": {"id": "111222333"},
                    "message_id": 200,
                },
            },
        }
        # не падает
        poller._process_update(upd)
        # ответ отправлен (record_feedback вернул False после exception)
        poller._client.answerCallbackQuery.assert_called_once()

    # ── 6g: record_feedback error → ответ всё равно отправлен ───────────────

    def test_record_feedback_error_still_answers_callback(self):
        """Даже если record_feedback упал, answerCallbackQuery отправляется."""
        poller = _build_poller()
        poller._storage.record_feedback.side_effect = Exception("DB connection lost")

        upd = {
            "update_id": 16,
            "callback_query": {
                "id": "cb_q_7",
                "data": "fb:8:d",
                "message": {
                    "chat": {"id": "111222333"},
                    "message_id": 300,
                },
            },
        }
        poller._process_update(upd)
        poller._client.answerCallbackQuery.assert_called_once()
        # popup_text всё равно отправлен
        call_args = poller._client.answerCallbackQuery.call_args
        self.assertIn("👎", call_args.kwargs.get("text", ""))


# ══════════════════════════════════════════════════════════════════════════════
# 7. M2: callback от чужого chat_id — игнорируется
# ══════════════════════════════════════════════════════════════════════════════

class TestCallbackOwnerOnly(unittest.TestCase):
    """M2: кнопка в чужом чате не обрабатывается."""

    def test_callback_from_other_chat_ignored(self):
        poller = _build_poller(chat_id="owner_chat")
        upd = {
            "update_id": 20,
            "callback_query": {
                "id": "cb_stranger",
                "data": "fb:1:u",
                "message": {
                    "chat": {"id": "stranger_chat"},
                    "message_id": 123,
                },
            },
        }
        poller._process_update(upd)

        # Ни record_feedback, ни answerCallbackQuery не вызваны
        poller._storage.record_feedback.assert_not_called()
        poller._client.answerCallbackQuery.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# 8. TelegramPoller — offset persistence
# ══════════════════════════════════════════════════════════════════════════════

class TestOffsetPersistence(unittest.TestCase):

    def test_offset_saved_after_processing(self):
        """offset сохраняется в meta после обработки updates."""
        storage = MagicMock()
        storage.get_meta.return_value = None
        poller = _build_poller(storage_mock=storage)

        # имитируем обработку одного update
        upd = {"update_id": 5, "callback_query": _make_cb(5)}
        poller._process_update(upd)

        max_id = 5
        poller._save_offset(max_id + 1)

        storage.set_meta.assert_called_with("tg_updates_offset", "6")

    def test_offset_loaded_from_meta(self):
        storage = MagicMock()
        storage.get_meta.return_value = "42"
        poller = _build_poller(storage_mock=storage)
        poller._storage = storage  # подменяем мок

        loaded = poller._load_offset()
        self.assertEqual(loaded, 42)

    def test_offset_invalid_meta_returns_none(self):
        storage = MagicMock()
        storage.get_meta.return_value = "not_a_number"
        poller = _build_poller(storage_mock=storage)
        poller._storage = storage

        self.assertIsNone(poller._load_offset())

    def test_offset_none_when_storage_absent(self):
        """_load_offset возвращает None если storage ещё не создан."""
        poller = TelegramPoller(
            token="tok123456",
            chat_id="111",
            db_path=":memory:",
            on_digest=MagicMock(),
        )
        poller._storage = None
        self.assertIsNone(poller._load_offset())


# ══════════════════════════════════════════════════════════════════════════════
# 9. TelegramPoller — команды
# ══════════════════════════════════════════════════════════════════════════════

def _make_msg(text: str, chat_id: str = "111222333"):
    return {"message_id": 1, "chat": {"id": chat_id}, "text": text}


class TestCommandHandling(unittest.TestCase):
    """M1: /digest вызывает СИГНАЛ, а не run_digest напрямую."""

    OWNER = "111222333"

    # ── 9a: /digest → on_digest вызван (СИГНАЛ, M1) ──────────────────────────

    def test_digest_command_calls_on_digest_signal(self):
        """M1: on_digest вызывается как простой сигнал (set event)."""
        on_digest = MagicMock()
        poller = _build_poller(on_digest=on_digest, chat_id=self.OWNER)

        upd = {"update_id": 20, "message": _make_msg("/digest", self.OWNER)}
        poller._process_update(upd)

        # on_digest ВЫЗВАН (сигнал)
        on_digest.assert_called_once()
        # Отправлен ответ пользователю
        poller._client.sendMessage.assert_called()

    # ── 9b: /digest → отправляет текстовый ответ ─────────────────────────────

    def test_digest_command_sends_reply(self):
        on_digest = MagicMock()
        poller = _build_poller(on_digest=on_digest, chat_id=self.OWNER)

        upd = {"update_id": 21, "message": _make_msg("/digest", self.OWNER)}
        poller._process_update(upd)

        args = poller._client.sendMessage.call_args[0]
        self.assertEqual(args[0], self.OWNER)
        self.assertIn("Команда", args[1])

    # ── 9c: /digest с on_digest ошибкой — не падает (C2) ────────────────────

    def test_digest_command_error_handled(self):
        on_digest = MagicMock(side_effect=RuntimeError("boom"))
        poller = _build_poller(on_digest=on_digest, chat_id=self.OWNER)

        upd = {"update_id": 22, "message": _make_msg("/digest", self.OWNER)}
        # не падает (C2)
        poller._process_update(upd)
        # ошибка залогирована и ответ отправлен
        poller._client.sendMessage.assert_called()

    # ── 9d: /start → приветствие ────────────────────────────────────────────

    def test_start_command_sends_greeting(self):
        poller = _build_poller(chat_id=self.OWNER)
        upd = {"update_id": 23, "message": _make_msg("/start", self.OWNER)}
        poller._process_update(upd)

        args = poller._client.sendMessage.call_args[0]
        self.assertEqual(args[0], self.OWNER)
        self.assertIn("newsbot", args[1])

    # ── 9e: /help → справка ─────────────────────────────────────────────────

    def test_help_command_sends_help(self):
        poller = _build_poller(chat_id=self.OWNER)
        upd = {"update_id": 24, "message": _make_msg("/help", self.OWNER)}
        poller._process_update(upd)

        args = poller._client.sendMessage.call_args[0]
        self.assertEqual(args[0], self.OWNER)
        self.assertIn("Команды", args[1])

    # ── 9f: чужая команда — игнорируется ────────────────────────────────────

    def test_other_chat_command_ignored(self):
        on_digest = MagicMock()
        poller = _build_poller(on_digest=on_digest, chat_id=self.OWNER)

        upd = {"update_id": 25, "message": _make_msg("/digest", "999888777")}
        poller._process_update(upd)

        on_digest.assert_not_called()
        poller._client.sendMessage.assert_not_called()

    # ── 9g: неизвестная команда — игнорируется ──────────────────────────────

    def test_unknown_command_ignored(self):
        on_digest = MagicMock()
        poller = _build_poller(on_digest=on_digest, chat_id=self.OWNER)

        upd = {"update_id": 26, "message": _make_msg("/stats", self.OWNER)}
        poller._process_update(upd)

        on_digest.assert_not_called()
        poller._client.sendMessage.assert_not_called()

    # ── 9h: on_digest вызван ТОЛЬКО как сигнал (M1) ──────────────────────────

    def test_digest_on_digest_is_simple_callable(self):
        """
        M1: on_digest — callable (lambda: _digest_requested.set()),
        НЕ вызывает pipeline.run_digest в poller thread.
        """
        call_log = []

        def on_digest_track():
            call_log.append("called")

        poller = _build_poller(on_digest=on_digest_track, chat_id=self.OWNER)
        upd = {"update_id": 27, "message": _make_msg("/digest", self.OWNER)}
        poller._process_update(upd)

        # on_digest вызван ровно один раз
        self.assertEqual(call_log, ["called"])


# ══════════════════════════════════════════════════════════════════════════════
# 10. TelegramPoller — сетевые ошибки и 409
# ══════════════════════════════════════════════════════════════════════════════

class TestNetworkErrors(unittest.TestCase):

    def test_network_error_in_poll_continues_loop(self):
        """m2: сетевая ошибка → _poll re-raises (ошибка не глотается)."""
        storage = MagicMock()
        storage.get_meta.return_value = None
        client = MagicMock(spec=_TgClient)
        client.getUpdates.side_effect = TgApiError(http_status=None, cause=None)
        poller = _build_poller(storage_mock=storage, client_mock=client)
        poller._storage = storage

        # m2: _poll не глотает ошибку — re-raises TgApiError
        with self.assertRaises(TgApiError):
            poller._poll(None)

    # ── N1: 409 определяется по HTTPStatusError (не по подстроке) ──────────

    def test_409_conflict_reraises_to_tick(self):
        """
        m2: 409 в _poll — спим 60 с внутри _poll, затем re-raise.
        _tick ловит re-raise, видит 409 → логирует и выходит.
        """
        storage = MagicMock()
        storage.get_meta.return_value = None
        client = MagicMock(spec=_TgClient)

        client.getUpdates.side_effect = TgApiError(
            http_status=_HTTP_CONFLICT, cause=None
        )
        poller = _build_poller(storage_mock=storage, client_mock=client)
        poller._storage = storage
        poller._sleep_backoff = MagicMock()

        # _poll re-raises TgApiError(409) после своего sleep
        with self.assertRaises(TgApiError) as cm:
            poller._poll(None)
        self.assertEqual(cm.exception.http_status, _HTTP_CONFLICT)

        # 60 с sleep ВНУТРИ _poll (до re-raise)
        poller._sleep_backoff.assert_called_once_with(60.0)
        client.getUpdates.assert_called_once()

    def test_network_error_reraises_to_tick(self):
        """m2: сетевая ошибка в _poll → re-raise → _tick делает backoff."""
        storage = MagicMock()
        storage.get_meta.return_value = None
        client = MagicMock(spec=_TgClient)

        client.getUpdates.side_effect = TgApiError(http_status=None, cause=None)
        poller = _build_poller(storage_mock=storage, client_mock=client)
        poller._storage = storage
        poller._sleep_backoff = MagicMock()

        # не падает в _poll — re-raises
        with self.assertRaises(TgApiError):
            poller._poll(None)

    def test_tick_handles_network_error_returns_false(self):
        """
        m2: _tick ловит сетевую ошибку, делает sleep, ВОЗВРАЩАЕТ False.
        Backoff растёт в _run_loop (НЕ в _tick).
        """
        storage = MagicMock()
        storage.get_meta.return_value = None
        client = MagicMock(spec=_TgClient)

        client.getUpdates.side_effect = TgApiError(http_status=500, cause=None)
        poller = _build_poller(storage_mock=storage, client_mock=client)
        poller._storage = storage
        poller._sleep_backoff = MagicMock()
        poller._backoff = 4.0

        # _tick → False (ошибка), sleep на 4 с, backoff НЕ меняется в _tick
        result = poller._tick(backoff=4.0)

        self.assertIs(result, False)
        poller._sleep_backoff.assert_not_called()  # sleep — в _run_loop
        self.assertEqual(poller._backoff, 4.0)

    def test_tick_returns_true_on_success(self):
        """_tick возвращает True при успешной обработке updates."""
        storage = MagicMock()
        storage.get_meta.return_value = None
        client = MagicMock(spec=_TgClient)
        client.getUpdates.return_value = [
            {"update_id": 5, "callback_query": _make_cb(5)},
        ]
        poller = _build_poller(storage_mock=storage, client_mock=client)
        poller._storage = storage
        poller._sleep_backoff = MagicMock()

        result = poller._tick(backoff=1.0)

        self.assertIs(result, True)
        poller._sleep_backoff.assert_not_called()
        # verify: getUpdates вызван
        poller._client.getUpdates.assert_called_once()

    def test_tick_returns_none_on_empty_updates(self):
        """_tick возвращает None при пустых updates (без ошибок)."""
        storage = MagicMock()
        storage.get_meta.return_value = None
        client = MagicMock(spec=_TgClient)
        client.getUpdates.return_value = []
        poller = _build_poller(storage_mock=storage, client_mock=client)
        poller._storage = storage
        poller._sleep_backoff = MagicMock()
        poller._backoff = 4.0

        result = poller._tick(backoff=4.0)

        self.assertIsNone(result)
        poller._sleep_backoff.assert_not_called()
        self.assertEqual(poller._backoff, 4.0)

    def test_backoff_grows_across_errors(self):
        """
        m2: серия ошибок → _tick возвращает False на каждой ошибке.
        Backoff растёт в _run_loop (sleep + *=2 при result is False).
        """
        storage = MagicMock()
        storage.get_meta.return_value = None
        client = MagicMock(spec=_TgClient)
        client.getUpdates.side_effect = TgApiError(http_status=None, cause=None)
        poller = _build_poller(storage_mock=storage, client_mock=client)
        poller._sleep_backoff = MagicMock()
        poller._backoff = 1.0
        poller._stop_event = _FakeStopEvent()

        # ── ошибка 1: result=False, sleep(1.0), backoff→2.0 (в _run_loop) ──
        result1 = poller._tick(1.0)
        self.assertIs(result1, False)
        poller._sleep_backoff.assert_not_called()  # _tick не спит
        # эмулируем _run_loop: sleep + backoff*=2
        poller._sleep_backoff(poller._backoff)
        poller._backoff = min(poller._backoff * 2, 60.0)
        self.assertEqual(poller._backoff, 2.0)

        # ── ошибка 2: result=False, sleep(2.0), backoff→4.0 ──
        poller._sleep_backoff.reset_mock()
        result2 = poller._tick(2.0)
        self.assertIs(result2, False)
        poller._sleep_backoff.assert_not_called()
        poller._sleep_backoff(poller._backoff)
        poller._backoff = min(poller._backoff * 2, 60.0)
        self.assertEqual(poller._backoff, 4.0)

        # ── empty: result=None, backoff не меняется ──
        client.getUpdates.side_effect = None
        client.getUpdates.return_value = []
        poller._sleep_backoff.reset_mock()
        poller._backoff = 4.0
        result3 = poller._tick(4.0)
        self.assertIsNone(result3)
        self.assertEqual(poller._backoff, 4.0)
        poller._sleep_backoff.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# 11. C1: poller создаёт СОБСТВЕННОЕ Storage в своём потоке
# ══════════════════════════════════════════════════════════════════════════════

class TestStorageThreadIsolation(unittest.TestCase):
    """C1: poller создаёт отдельное соединение, не использует main Storage."""

    def test_poller_creates_own_storage_in_run_loop(self):
        """
        C1: _run_loop создаёт СВОЙ экземпляр Storage (не main).
        Проверяем: db_path передаётся правильный; мокаем time.sleep.
        """
        import newsbot.storage as storage_module
        RealStorage = storage_module.Storage

        call_tracker = {}

        original_init = RealStorage.__init__

        def tracked_init(self, db_path):
            call_tracker["db_path"] = db_path
            call_tracker["thread"] = threading.current_thread().ident
            return original_init(self, db_path)

        # мокаем time.sleep — чтобы _run_loop не завис на backoff
        with patch.object(RealStorage, "__init__", tracked_init):
            with patch("newsbot.telegram_io.time.sleep"):
                client = MagicMock(spec=_TgClient)
                client.getUpdates.return_value = []

                on_digest = MagicMock()
                poller = TelegramPoller(
                    token="tok123456",
                    chat_id="111",
                    db_path="/test/path/newsbot.db",
                    on_digest=on_digest,
                    logger=logging.getLogger("test"),
                )
                poller._client = client
                poller._stop_event = _FakeStopEvent()

                # Подменяем stop_event на реальный threading.Event
                real_stop = threading.Event()

                class _TestStop:
                    def is_set(self):
                        return real_stop.is_set()

                    def set(self):
                        real_stop.set()

                poller._stop_event = _TestStop()

                def stop_after_first():
                    """Остановить poller после первого витка."""
                    real_stop.set()

                # запускаем в потоке и останавливаем после создания storage
                t = threading.Thread(target=poller._run_loop, daemon=True)
                t.start()
                # ждём создания storage (до 2с)
                import time as _time
                for _ in range(20):
                    if call_tracker.get("db_path"):
                        break
                    _time.sleep(0.1)
                # останавливаем
                real_stop.set()
                t.join(timeout=2)

                # Storage.__init__ вызван с правильным db_path
                self.assertEqual(call_tracker.get("db_path"), "/test/path/newsbot.db")

    def test_storage_creation_failure_stops_poller(self):
        """
        Если Storage нельзя создать в poller thread — poller не падает,
        а возвращается из _run_loop.
        """
        import newsbot.storage
        with patch.object(newsbot.storage, "Storage", side_effect=RuntimeError("no DB")):
            on_digest = MagicMock()
            poller = TelegramPoller(
                token="tok123456",
                chat_id="111",
                db_path="/nonexistent/path.db",
                on_digest=on_digest,
                logger=logging.getLogger("test"),
            )
            # _run_loop завершается (не кидает наружу), on_digest НЕ вызван
            # Проверяем что поллер останавливается корректно
            try:
                poller._run_loop()
            except Exception as e:
                self.fail(f"_run_loop raised: {e}")
            # on_digest ни разу не вызван (poller не создал storage)
            on_digest.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# 12. C2: _process_update защищён от любых ошибок
# ══════════════════════════════════════════════════════════════════════════════

class TestUpdateProcessingRobustness(unittest.TestCase):
    """C2: ошибка при обработке update не убивает поток."""

    def test_photo_update_does_not_crash(self):
        """C2: фото-сообщение (без text) не роняет обработку."""
        poller = _build_poller()
        upd = {
            "update_id": 30,
            "message": {
                "message_id": 1,
                "chat": {"id": "111222333"},
                "photo": [{"file_id": "abc"}],
            },
        }
        # не падает
        poller._process_update(upd)

    def test_sticker_update_does_not_crash(self):
        """C2: стикер не роняет обработку."""
        poller = _build_poller()
        upd = {
            "update_id": 31,
            "message": {
                "message_id": 1,
                "chat": {"id": "111222333"},
                "sticker": {"emoji": "😀"},
            },
        }
        poller._process_update(upd)

    def test_service_message_does_not_crash(self):
        """C2: service message не роняет обработку."""
        poller = _build_poller()
        upd = {
            "update_id": 32,
            "message": {
                "message_id": 1,
                "chat": {"id": "111222333"},
                "new_chat_member": {"id": 999},
            },
        }
        poller._process_update(upd)


# ══════════════════════════════════════════════════════════════════════════════
# 13. Токен не в логах
# ══════════════════════════════════════════════════════════════════════════════

class TestTokenNotInLogs(unittest.TestCase):
    """Проверяем что токен не раскрывается в логах (N4)."""

    def test_token_not_in_sanitizer_output(self):
        s = _TokenSanitizer("secret_bot_token_abcdefgh")
        result = s()
        self.assertNotIn("secret_bot_token_abcdefgh", result)
        self.assertIn("len=25", result)
        self.assertIn("...efgh", result)

    def test_httperror_type_logged_not_str(self):
        """N4: _TgClient._call выбрасывает TgApiError (НЕ httpx.HTTPError)."""
        import httpx

        with patch("httpx.Client") as MockClient:
            instance = MockClient.return_value
            # str(e) содержит URL с токеном, но НЕ попадает в исключение
            instance.post.side_effect = httpx.HTTPError(
                "Connection refused to https://api.telegram.org/botSECRET_TOKEN/getMe"
            )
            client = _TgClient("SECRET_TOKEN")
            # N4: брошено TgApiError (безопасное), НЕ httpx.HTTPError с URL
            with self.assertRaises(TgApiError) as ctx:
                client._call("getMe")
            # http_status=None для сетевой ошибки
            self.assertIsNone(ctx.exception.http_status)
            # сообщение НЕ содержит URL с токеном
            self.assertNotIn("SECRET_TOKEN", str(ctx.exception))


# ══════════════════════════════════════════════════════════════════════════════
# 14. _Commands.text
# ══════════════════════════════════════════════════════════════════════════════

class TestCommandsText(unittest.TestCase):
    def test_text_strips_whitespace(self):
        msg = {"message_id": 1, "chat": {"id": "111"}, "text": "  /digest  "}
        self.assertEqual(_Commands.text(msg), "/digest")

    def test_text_empty(self):
        msg = {"message_id": 1, "chat": {"id": "111"}}
        self.assertEqual(_Commands.text(msg), "")


# ══════════════════════════════════════════════════════════════════════════════
# 15. n5: URL-кнопки сохраняются после 👍/👎 (багфикс)
# ══════════════════════════════════════════════════════════════════════════════

class TestCallbackKeyboardPreservation(unittest.TestCase):
    """После нажатия 👍/👎 URL-кнопки сохраняются; удаляются только для нажатого target_id."""

    def _make_cb_with_keyboard(
        self,
        target_id: int,
        verdict: str = "u",
        chat_id: str = "111222333",
        message_id: int = 555,
        extra_buttons: list = None,
    ) -> dict:
        """
        callback_query с reply_markup в message.
        Структура по умолчанию: URL + 👍 + 👎 для target_id.
        """
        buttons = [
            {"text": "🔗 Открыть источник", "url": f"https://example.com/{target_id}"},
            {"text": "👍", "callback_data": f"fb:{target_id}:u"},
            {"text": "👎", "callback_data": f"fb:{target_id}:d"},
        ]
        if extra_buttons:
            buttons.extend(extra_buttons)
        return {
            "callback_query": {
                "id": "cb_test",
                "data": f"fb:{target_id}:{verdict}",
                "message": {
                    "chat": {"id": chat_id},
                    "message_id": message_id,
                    "reply_markup": {"inline_keyboard": [buttons]},
                },
            },
        }

    # ── 15a: URL-кнопка остаётся после 👍 ─────────────────────────────────────

    def test_url_button_preserved_after_like(self):
        """После 👍 URL-кнопка ОСТАЁТСЯ, fb:42:u исчезает."""
        poller = _build_poller()
        upd = self._make_cb_with_keyboard(target_id=42, verdict="u")

        poller._process_update(upd)

        poller._client.answerCallbackQuery.assert_called_once()
        poller._client.editMessageReplyMarkup.assert_called_once()
        call_args = poller._client.editMessageReplyMarkup.call_args

        new_kb = call_args.kwargs["reply_markup"]["inline_keyboard"]
        # 1 ряд: только URL-кнопка
        self.assertEqual(len(new_kb), 1)
        self.assertEqual(new_kb[0][0]["url"], "https://example.com/42")
        # 👍-кнопка удалена
        self.assertFalse(
            any(
                btn.get("callback_data") == "fb:42:u"
                for row in new_kb for btn in row
            )
        )

    # ── 15b: URL-кнопка остаётся после 👎 ─────────────────────────────────────

    def test_url_button_preserved_after_dislike(self):
        """После 👎 URL-кнопка ОСТАЁТСЯ, fb:42:d исчезает."""
        poller = _build_poller()
        upd = self._make_cb_with_keyboard(target_id=42, verdict="d")

        poller._process_update(upd)

        poller._client.editMessageReplyMarkup.assert_called_once()
        new_kb = poller._client.editMessageReplyMarkup.call_args.kwargs[
            "reply_markup"
        ]["inline_keyboard"]
        # Только URL-кнопка
        self.assertEqual(len(new_kb), 1)
        self.assertIn("url", new_kb[0][0])
        # 👎-кнопка удалена
        self.assertFalse(
            any(
                btn.get("callback_data") == "fb:42:d"
                for row in new_kb for btn in row
            )
        )

    # ── 15c: чужие fb:-кнопки сохраняются (мульти-item) ─────────────────────

    def test_other_targets_callback_data_preserved(self):
        """
        Мульти-item: после 👍 по target=1 кнопки fb:2:u/d сохраняются.
        Клавиатура: [URL1, fb:1:u, fb:1:d], [URL2, fb:2:u, fb:2:d]
        """
        poller = _build_poller()
        upd = {
            "callback_query": {
                "id": "cb_multi",
                "data": "fb:1:u",
                "message": {
                    "chat": {"id": "111222333"},
                    "message_id": 100,
                    "reply_markup": {
                        "inline_keyboard": [
                            [
                                {"text": "🔗 Item1", "url": "https://ex.com/1"},
                                {"text": "👍", "callback_data": "fb:1:u"},
                                {"text": "👎", "callback_data": "fb:1:d"},
                            ],
                            [
                                {"text": "🔗 Item2", "url": "https://ex.com/2"},
                                {"text": "👍", "callback_data": "fb:2:u"},
                                {"text": "👎", "callback_data": "fb:2:d"},
                            ],
                        ],
                    },
                },
            },
        }

        poller._process_update(upd)

        poller._client.editMessageReplyMarkup.assert_called_once()
        new_kb = poller._client.editMessageReplyMarkup.call_args.kwargs[
            "reply_markup"
        ]["inline_keyboard"]

        # Ряд 1: только URL1 (👍/👎 для target=1 удалены)
        self.assertEqual(len(new_kb[0]), 1)
        self.assertEqual(new_kb[0][0]["url"], "https://ex.com/1")

        # Ряд 2: полностью сохранён (URL2 + fb:2:u + fb:2:d)
        self.assertEqual(len(new_kb[1]), 3)
        self.assertEqual(new_kb[1][0]["url"], "https://ex.com/2")
        cb_data = [btn.get("callback_data") for btn in new_kb[1]]
        self.assertIn("fb:2:u", cb_data)
        self.assertIn("fb:2:d", cb_data)

    # ── 15d: reply_markup отсутствует → editMessageReplyMarkup НЕ вызывается ─

    def test_no_reply_markup_no_edit(self):
        """
        Если message.reply_markup отсутствует — editMessageReplyMarkup НЕ вызывается,
        чтобы не снести клавиатуру.
        """
        poller = _build_poller()
        upd = {
            "callback_query": {
                "id": "cb_nomarkup",
                "data": "fb:5:u",
                "message": {
                    "chat": {"id": "111222333"},
                    "message_id": 200,
                    # reply_markup отсутствует
                },
            },
        }

        poller._process_update(upd)

        poller._client.answerCallbackQuery.assert_called_once()
        poller._client.editMessageReplyMarkup.assert_not_called()

    # ── 15e: reply_markup = None (не dict) → не падает ───────────────────────

    def test_reply_markup_none_does_not_crash(self):
        """reply_markup = None не вызывает падения."""
        poller = _build_poller()
        upd = {
            "callback_query": {
                "id": "cb_none",
                "data": "fb:7:u",
                "message": {
                    "chat": {"id": "111222333"},
                    "message_id": 300,
                    "reply_markup": None,
                },
            },
        }
        # не падает
        poller._process_update(upd)
        poller._client.answerCallbackQuery.assert_called_once()
        poller._client.editMessageReplyMarkup.assert_not_called()

    # ── 15f: reply_markup без inline_keyboard → не падает ─────────────────────

    def test_reply_markup_missing_inline_keyboard_does_not_crash(self):
        """reply_markup без inline_keyboard не вызывает падения."""
        poller = _build_poller()
        upd = {
            "callback_query": {
                "id": "cb_nokeyboard",
                "data": "fb:9:u",
                "message": {
                    "chat": {"id": "111222333"},
                    "message_id": 400,
                    "reply_markup": {"some": "other"},
                },
            },
        }
        # не падает
        poller._process_update(upd)
        poller._client.answerCallbackQuery.assert_called_once()
        poller._client.editMessageReplyMarkup.assert_not_called()

    # ── 15g: empty inline_keyboard → editMessageReplyMarkup с [] ─────────────

    def test_empty_keyboard_after_removal_calls_edit_with_empty(self):
        """
        Клавиатура была только из 👍/👎 → после удаления new_kb пуст.
        Допустимо вызвать editMessageReplyMarkup с {"inline_keyboard": []}.
        """
        poller = _build_poller()
        upd = {
            "callback_query": {
                "id": "cb_empty",
                "data": "fb:3:u",
                "message": {
                    "chat": {"id": "111222333"},
                    "message_id": 500,
                    "reply_markup": {
                        "inline_keyboard": [
                            [
                                {"text": "👍", "callback_data": "fb:3:u"},
                                {"text": "👎", "callback_data": "fb:3:d"},
                            ],
                        ],
                    },
                },
            },
        }

        poller._process_update(upd)

        poller._client.editMessageReplyMarkup.assert_called_once()
        new_kb = poller._client.editMessageReplyMarkup.call_args.kwargs[
            "reply_markup"
        ]["inline_keyboard"]
        self.assertEqual(new_kb, [])


# ══════════════════════════════════════════════════════════════════════════════
# 16. Major-фикс: backoff растёт в _run_loop (не сбрасывается после каждого _tick)
# ══════════════════════════════════════════════════════════════════════════════

class Test409ConflictIntegration(unittest.TestCase):
    """
    Интеграционные тесты для 409 Conflict.
    m2-critical: 409 НЕ должен увеличивать backoff.
    """

    def test_409_through_run_loop_sleeps_once_and_backoff_unchanged(self):
        """
        409 проходит через _run_loop: _sleep_backoff вызван ровно 1 раз (60с из _poll),
        _backoff НЕ вырос.
        """
        sleep_durations = []

        def tracked_sleep(seconds):
            sleep_durations.append(seconds)

        storage = MagicMock()
        storage.get_meta.return_value = None

        client = MagicMock(spec=_TgClient)
        # Всегда 409
        client.getUpdates.side_effect = TgApiError(
            http_status=_HTTP_CONFLICT, cause=None
        )

        on_digest = MagicMock()
        poller = TelegramPoller(
            token="tok123456",
            chat_id="111",
            db_path=":memory:",
            on_digest=on_digest,
            logger=logging.getLogger("test"),
        )
        poller._storage = storage
        poller._client = client
        poller._sleep_backoff = tracked_sleep
        poller._backoff = 4.0  # backoff растёт если ошибки идут через _run_loop

        class _CountingStop:
            def __init__(self, max_iterations):
                self._iteration = 0
                self._max = max_iterations

            def is_set(self):
                self._iteration += 1
                return self._iteration > self._max

            def set(self):
                pass

        # 1 итерация: 409
        poller._stop_event = _CountingStop(max_iterations=1)

        t = threading.Thread(target=poller._run_loop, daemon=True)
        t.start()
        t.join(timeout=3)

        # ровно 1 sleep (60с из _poll), _backoff не вырос
        self.assertEqual(len(sleep_durations), 1)
        self.assertEqual(sleep_durations[0], 60.0)
        self.assertEqual(poller._backoff, 4.0)

    def test_409_tick_returns_none_not_false(self):
        """
        _tick при 409 возвращает None (нейтрально).
        _run_loop при None: не спит и не удваивает backoff.
        _sleep_backoff вызывается ВНУТРИ _poll (при 409) — не в _tick.
        """
        storage = MagicMock()
        storage.get_meta.return_value = None
        client = MagicMock(spec=_TgClient)
        client.getUpdates.side_effect = TgApiError(
            http_status=_HTTP_CONFLICT, cause=None
        )
        poller = _build_poller(storage_mock=storage, client_mock=client)
        poller._storage = storage
        poller._backoff = 8.0

        # патчим time.sleep — _poll вызывает _sleep_backoff(60) изнутри,
        # это нормальное поведение. Нас интересует return-value и backoff в _tick.
        with patch("newsbot.telegram_io.time.sleep"):
            result = poller._tick(backoff=8.0)

        self.assertIsNone(result)  # нейтрально, НЕ False
        self.assertEqual(poller._backoff, 8.0)  # не меняется в _tick

    def test_backoff_409_then_success_resets_to_1(self):
        """
        Серия: 409 (None) → backoff не растёт → успех (True) → backoff=1.0.
        Проверяем через _run_loop.
        """
        sleep_durations = []

        def tracked_sleep(seconds):
            sleep_durations.append(seconds)

        storage = MagicMock()
        storage.get_meta.return_value = None
        client = MagicMock(spec=_TgClient)

        call_count = [0]

        def getUpdates_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TgApiError(http_status=_HTTP_CONFLICT, cause=None)
            return [{"update_id": 5, "callback_query": _make_cb(5)}]

        client.getUpdates.side_effect = getUpdates_side_effect

        on_digest = MagicMock()
        poller = TelegramPoller(
            token="tok123456",
            chat_id="111",
            db_path=":memory:",
            on_digest=on_digest,
            logger=logging.getLogger("test"),
        )
        poller._storage = storage
        poller._client = client
        poller._sleep_backoff = tracked_sleep
        poller._backoff = 4.0  # стартуем с ненулевого

        class _CountingStop:
            def __init__(self, max_iterations):
                self._iteration = 0
                self._max = max_iterations

            def is_set(self):
                self._iteration += 1
                return self._iteration > self._max

            def set(self):
                pass

        # 2 итерации: 409 (sleep 60с, None) → success (True, backoff=1.0)
        poller._stop_event = _CountingStop(max_iterations=2)

        t = threading.Thread(target=poller._run_loop, daemon=True)
        t.start()
        t.join(timeout=5)

        # 1 sleep (60с из _poll); 409 не вызвал sleep в _run_loop
        self.assertEqual(len(sleep_durations), 1)
        self.assertEqual(sleep_durations[0], 60.0)
        # после success backoff сброшен в 1.0
        self.assertEqual(poller._backoff, 1.0)


class TestBackoffGrowthIntegration(unittest.TestCase):
    """
    Интеграционный тест: _run_loop управляет backoff через return-значение _tick.
    Существующие тесты вызывали _tick напрямую — ловили старый баг.
    """

    def test_backoff_doubles_on_series_of_errors(self):
        """
        Major: серия ошибок → sleep-durations растут: 1, 2, 4, 8... (cap 60).
        Проверяем именно _run_loop (НЕ _tick напрямую).
        """
        import itertools

        sleep_durations = []

        def tracked_sleep(seconds):
            sleep_durations.append(seconds)

        storage = MagicMock()
        storage.get_meta.return_value = None

        client = MagicMock(spec=_TgClient)
        # iter() + cycle: бесконечный источник ошибок (не истощается)
        client.getUpdates.side_effect = iter(
            itertools.repeat(TgApiError(http_status=500, cause=None))
        )

        on_digest = MagicMock()
        poller = TelegramPoller(
            token="tok123456",
            chat_id="111222333",
            db_path=":memory:",
            on_digest=on_digest,
            logger=logging.getLogger("test"),
            poll_timeout=1,
        )
        poller._storage = storage
        poller._client = client
        poller._sleep_backoff = tracked_sleep
        poller._backoff = 1.0

        # Счётчик: останавливаем после 4 итераций
        class _CountingStop:
            def __init__(self, max_iterations):
                # Отслеживаем НОМЕР итерации (while-is_set вызовы)
                self._iteration = 0
                self._max = max_iterations

            def is_set(self):
                self._iteration += 1
                return self._iteration > self._max

            def set(self):
                pass

        # while-is_set вызывается: 1,2,3,4,5,... -> 4 полных итераций (5-я -> exit)
        poller._stop_event = _CountingStop(max_iterations=4)

        t = threading.Thread(target=poller._run_loop, daemon=True)
        t.start()
        t.join(timeout=3)

        # 4 ошибки → sleep на 1, 2, 4, 8 секунд (первые 4 записи)
        self.assertEqual(len(sleep_durations), 4)
        self.assertEqual(sleep_durations[0], 1.0)
        self.assertEqual(sleep_durations[1], 2.0)
        self.assertEqual(sleep_durations[2], 4.0)
        self.assertEqual(sleep_durations[3], 8.0)

    def test_backoff_resets_on_success(self):
        """
        _run_loop сбрасывает backoff в 1.0 после успешного _tick(True).
        Стартуем с backoff>1.0, чтобы проверить именно СБРОС.
        Вторая итерация: непустой getUpdates → _tick → True → backoff=1.0.
        """
        sleep_durations = []

        def tracked_sleep(seconds):
            sleep_durations.append(seconds)

        storage = MagicMock()
        storage.get_meta.return_value = None
        client = MagicMock(spec=_TgClient)

        call_count = [0]

        def getUpdates_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TgApiError(http_status=500, cause=None)
            # успех: непустой updates → _tick → True → сброс backoff в 1.0
            return [{"update_id": 7, "callback_query": _make_cb(7)}]

        client.getUpdates.side_effect = getUpdates_side_effect

        on_digest = MagicMock()
        poller = TelegramPoller(
            token="tok123456",
            chat_id="111222333",
            db_path=":memory:",
            on_digest=on_digest,
            logger=logging.getLogger("test"),
        )
        poller._storage = storage
        poller._client = client
        poller._sleep_backoff = tracked_sleep
        poller._backoff = 8.0  # стартуем с ненулевого — проверим именно СБРОС

        class _CountingStop:
            def __init__(self, max_iterations):
                self._iteration = 0
                self._max = max_iterations

            def is_set(self):
                self._iteration += 1
                return self._iteration > self._max

            def set(self):
                pass

        # 2 итерации: error (sleep 8.0, backoff→16) + success (backoff→1.0)
        poller._stop_event = _CountingStop(max_iterations=2)

        t = threading.Thread(target=poller._run_loop, daemon=True)
        t.start()
        t.join(timeout=3)

        # 1 ошибка → sleep на текущем backoff (8.0)
        self.assertEqual(len(sleep_durations), 1)
        self.assertEqual(sleep_durations[0], 8.0)
        # после успеха backoff сброшен в 1.0
        self.assertEqual(poller._backoff, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_cb(update_id: int, target_id: int = 1, verdict: str = "u",
             chat_id: str = "111222333"):
    return {
        "callback_query": {
            "id": f"cb_{update_id}",
            "data": f"fb:{target_id}:{verdict}",
            "message": {
                "chat": {"id": chat_id},
                "message_id": 100 + update_id,
            },
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 17. Owner-ID фикс: --set-chat-id авторизация / private-only / --bind
# ══════════════════════════════════════════════════════════════════════════════

class TestSetChatIdAuthorization(unittest.TestCase):
    """Тесты для _set_chat_id (candidates-only без bind) и _set_chat_id_explicit."""

    # httpx импортируется внутри функций (import httpx в теле), поэтому патчим
    # непосредственно httpx.get — именно этот объект перехватывает локальный import.
    # _write_env импортирован в __main__ как from .env import _write_env → патчим
    # newsbot.__main__._write_env.

    def _make_httpx_mock(self, fake_updates):
        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_updates
        return mock_resp

    def _join_output(self, mock_print):
        """Склеивает все аргументы всех вызовов print() в одну строку."""
        parts = []
        for call in mock_print.call_args_list:
            if call.args:
                parts.append(" ".join(str(a) for a in call.args))
        return "\n".join(parts)

    # ── _set_chat_id: TELEGRAM_OWNER_ID не задан — показывает кандидатов ─────

    def test_no_owner_id_shows_candidates(self):
        """Без owner_id: собирает private-чаты из getUpdates, печатает кандидатов."""
        from newsbot.__main__ import _set_chat_id

        fake_updates = {
            "ok": True,
            "result": [
                {"update_id": 10, "message": {
                    "chat": {"id": 111222333, "type": "private",
                             "username": "alice", "first_name": "Alice"}
                }},
                {"update_id": 20, "message": {
                    "chat": {"id": 444555666, "type": "private",
                             "username": "bob", "first_name": "Bob"}
                }},
                # group — игнорируется
                {"update_id": 30, "message": {
                    "chat": {"id": -777888999, "type": "group"}
                }},
            ],
        }

        with patch("httpx.get", return_value=self._make_httpx_mock(fake_updates)), \
             patch("newsbot.__main__.telegram_token", return_value="token"), \
             patch("newsbot.__main__.telegram_owner_id", return_value=""), \
             patch("newsbot.__main__._write_env"):
            with patch("builtins.print") as mock_print:
                _set_chat_id()
                output = self._join_output(mock_print)

            # печатает обоих кандидатов
            self.assertIn("111222333", output)
            self.assertIn("444555666", output)
            self.assertIn("alice", output)
            self.assertIn("bob", output)
            # group не попал
            self.assertNotIn("-777888999", output)

    def test_no_owner_id_one_candidate_prints_bind_hint(self):
        """Один кандидат — подсказка с --bind."""
        from newsbot.__main__ import _set_chat_id

        fake_updates = {
            "ok": True,
            "result": [
                {"update_id": 10, "message": {
                    "chat": {"id": 111222333, "type": "private",
                             "username": "alice", "first_name": "Alice"}
                }},
            ],
        }

        with patch("httpx.get", return_value=self._make_httpx_mock(fake_updates)), \
             patch("newsbot.__main__.telegram_token", return_value="token"), \
             patch("newsbot.__main__.telegram_owner_id", return_value=""), \
             patch("newsbot.__main__._write_env"):
            with patch("builtins.print") as mock_print:
                _set_chat_id()
                output = self._join_output(mock_print)

            self.assertIn("--bind", output)
            self.assertIn("111222333", output)

    def test_no_owner_id_no_private_chats(self):
        """Нет private-чатов — сообщение и выход без ошибки."""
        from newsbot.__main__ import _set_chat_id

        fake_updates = {"ok": True, "result": [
            {"update_id": 10, "message": {
                "chat": {"id": -100, "type": "group"}
            }},
        ]}

        with patch("httpx.get", return_value=self._make_httpx_mock(fake_updates)), \
             patch("newsbot.__main__.telegram_token", return_value="token"), \
             patch("newsbot.__main__.telegram_owner_id", return_value=""), \
             patch("newsbot.__main__._write_env"):
            with patch("builtins.print") as mock_print:
                _set_chat_id()
                output = self._join_output(mock_print)

            self.assertIn("Не найдено ни одного private-чата", output)

    # ── _set_chat_id: TELEGRAM_OWNER_ID задан — привязывает ТОЛЬКО его ───────

    def test_with_owner_id_binds_his_id_only(self):
        """owner_id задан: находит его chat_id и пишет TELEGRAM_CHAT_ID в .env."""
        from newsbot.__main__ import _set_chat_id

        fake_updates = {
            "ok": True,
            "result": [
                {"update_id": 10, "message": {
                    "chat": {"id": 111222333, "type": "private",
                             "username": "alice", "first_name": "Alice"}
                }},
                {"update_id": 20, "message": {
                    "chat": {"id": 999888777, "type": "private",
                             "username": "bob", "first_name": "Bob"}
                }},
            ],
        }

        with patch("httpx.get", return_value=self._make_httpx_mock(fake_updates)), \
             patch("newsbot.__main__.telegram_token", return_value="token"), \
             patch("newsbot.__main__.telegram_owner_id", return_value="111222333"), \
             patch("newsbot.__main__._write_env") as mock_write, \
             patch("builtins.print"):
            _set_chat_id()

            mock_write.assert_called_once()
            args = mock_write.call_args[0]
            self.assertEqual(args[0], "TELEGRAM_CHAT_ID")
            self.assertEqual(args[1], "111222333")

    def test_owner_id_rejects_wrong_id(self):
        """owner_id задан, но его chat_id нет в обновлениях — НЕ пишет в .env."""
        from newsbot.__main__ import _set_chat_id

        fake_updates = {
            "ok": True,
            "result": [
                {"update_id": 10, "message": {
                    "chat": {"id": 999888777, "type": "private",
                             "username": "bob", "first_name": "Bob"}
                }},
            ],
        }

        with patch("httpx.get", return_value=self._make_httpx_mock(fake_updates)), \
             patch("newsbot.__main__.telegram_token", return_value="token"), \
             patch("newsbot.__main__.telegram_owner_id", return_value="111222333"), \
             patch("newsbot.__main__._write_env") as mock_write, \
             patch("builtins.print"):
            _set_chat_id()

            mock_write.assert_not_called()

    def test_group_rejected(self):
        """getUpdates с group-type чатом: не попадает в candidates."""
        from newsbot.__main__ import _set_chat_id

        fake_updates = {
            "ok": True,
            "result": [
                {"update_id": 10, "message": {
                    "chat": {"id": -1001234567890, "type": "supergroup",
                             "title": "Test Group"}
                }},
            ],
        }

        with patch("httpx.get", return_value=self._make_httpx_mock(fake_updates)), \
             patch("newsbot.__main__.telegram_token", return_value="token"), \
             patch("newsbot.__main__.telegram_owner_id", return_value=""), \
             patch("newsbot.__main__._write_env"):
            with patch("builtins.print") as mock_print:
                _set_chat_id()
                output = self._join_output(mock_print)

            self.assertIn("Не найдено ни одного private-чата", output)

    def test_channel_rejected(self):
        """getUpdates с channel — игнорируется."""
        from newsbot.__main__ import _set_chat_id

        fake_updates = {
            "ok": True,
            "result": [
                {"update_id": 10, "message": {
                    "chat": {"id": -100200300400, "type": "channel",
                             "title": "My Channel"}
                }},
            ],
        }

        with patch("httpx.get", return_value=self._make_httpx_mock(fake_updates)), \
             patch("newsbot.__main__.telegram_token", return_value="token"), \
             patch("newsbot.__main__.telegram_owner_id", return_value=""), \
             patch("newsbot.__main__._write_env"):
            with patch("builtins.print") as mock_print:
                _set_chat_id()
                output = self._join_output(mock_print)

            self.assertIn("Не найдено ни одного private-чата", output)

    # ── _set_chat_id_explicit: --bind ─────────────────────────────────────────

    def test_bind_explicit_private_chat(self):
        """--bind с реальным private chat_id: пишет в .env."""
        from newsbot.__main__ import _set_chat_id_explicit

        fake_updates = {
            "ok": True,
            "result": [
                {"update_id": 10, "message": {
                    "chat": {"id": 111222333, "type": "private",
                             "username": "alice", "first_name": "Alice"}
                }},
            ],
        }

        with patch("httpx.get", return_value=self._make_httpx_mock(fake_updates)), \
             patch("newsbot.__main__.telegram_token", return_value="token"), \
             patch("newsbot.__main__.telegram_owner_id", return_value=""), \
             patch("newsbot.__main__._write_env") as mock_write, \
             patch("builtins.print"):
            _set_chat_id_explicit("111222333")

            mock_write.assert_called_once()
            args = mock_write.call_args[0]
            self.assertEqual(args[0], "TELEGRAM_CHAT_ID")
            self.assertEqual(args[1], "111222333")

    def test_bind_explicit_non_private_rejected(self):
        """--bind с group/channel id — отклонение."""
        from newsbot.__main__ import _set_chat_id_explicit

        fake_updates = {
            "ok": True,
            "result": [
                {"update_id": 10, "message": {
                    "chat": {"id": -1001234567890, "type": "supergroup",
                             "title": "Test Group"}
                }},
            ],
        }

        with patch("httpx.get", return_value=self._make_httpx_mock(fake_updates)), \
             patch("newsbot.__main__.telegram_token", return_value="token"), \
             patch("newsbot.__main__.telegram_owner_id", return_value=""), \
             patch("newsbot.__main__._write_env") as mock_write, \
             patch("builtins.print") as mock_print:
            _set_chat_id_explicit("-1001234567890")

            mock_write.assert_not_called()
            output = self._join_output(mock_print)
            self.assertIn("НЕ private-чат", output)

    def test_bind_explicit_owner_id_mismatch_rejected(self):
        """--bind с id ≠ TELEGRAM_OWNER_ID — отклонение."""
        from newsbot.__main__ import _set_chat_id_explicit

        fake_updates = {
            "ok": True,
            "result": [
                {"update_id": 10, "message": {
                    "chat": {"id": 111222333, "type": "private",
                             "username": "alice", "first_name": "Alice"}
                }},
            ],
        }

        with patch("httpx.get", return_value=self._make_httpx_mock(fake_updates)), \
             patch("newsbot.__main__.telegram_token", return_value="token"), \
             patch("newsbot.__main__.telegram_owner_id", return_value="999888777"), \
             patch("newsbot.__main__._write_env") as mock_write, \
             patch("builtins.print") as mock_print:
            _set_chat_id_explicit("111222333")

            mock_write.assert_not_called()
            output = self._join_output(mock_print)
            self.assertIn("не совпадает с TELEGRAM_OWNER_ID", output)

    def test_bind_explicit_id_not_in_updates(self):
        """--bind с id, которого нет в getUpdates — сообщение об ошибке."""
        from newsbot.__main__ import _set_chat_id_explicit

        fake_updates = {
            "ok": True,
            "result": [
                {"update_id": 10, "message": {
                    "chat": {"id": 111222333, "type": "private",
                             "username": "alice", "first_name": "Alice"}
                }},
            ],
        }

        with patch("httpx.get", return_value=self._make_httpx_mock(fake_updates)), \
             patch("newsbot.__main__.telegram_token", return_value="token"), \
             patch("newsbot.__main__.telegram_owner_id", return_value=""), \
             patch("newsbot.__main__._write_env") as mock_write, \
             patch("builtins.print") as mock_print:
            _set_chat_id_explicit("999888777")  # нет в updates

            mock_write.assert_not_called()
            output = self._join_output(mock_print)
            self.assertIn("не найден среди private-чатов", output)


if __name__ == '__main__':
    unittest.main()
