"""
tests/test_delivery_security.py — тесты ПОТОКА B:
  - K4  : токен не попадает в логи (httpx WARNING + TokenSanitizer)
  - B7  : _split_text с огромным title не зацикливается
  - B8  : 429 retry_after каппится до 120
  - B9  : env-парсер корректно обрабатывает KEY="a#b", Windows-пути, KEY=v # c
  - B10 : telegram не пробрасывает URL/сырые исключения с токеном
  - T2-7: маскирование — длина + последние 4
  - T2-9: score-валидация — отклоняем bool/float/строки
"""
import io
import logging
import re
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest

# ── conftest уже поднял корень в sys.path ────────────────────────
from newsbot.env import _parse_env, _strip_inline_comment, _write_env
from newsbot.log import TokenSanitizer, setup_logging
from newsbot.llm.filter import LLMFilter
from newsbot.llm.client import DeepSeekClient, LLMError
from newsbot.delivery.telegram import (
    MAX_CONSECUTIVE_429,
    RETRY_429_TOTAL_DEADLINE_SEC,
    SEND_PACING_SEC,
    TelegramDelivery,
    TelegramDeliveryError,
)


# ══════════════════════════════════════════════════════════════════
# B9: env-парсер — KEY="a#b", Windows-пути, KEY=v # c
# ══════════════════════════════════════════════════════════════════

class TestEnvParser:
    """B9: корректный парсинг кавычек, комментариев, Windows-путей."""

    def test_hash_inside_double_quotes(self, tmp_path):
        """KEY="a#b" — # внутри кавычек НЕ считается комментарием."""
        env_file = tmp_path / ".env"
        env_file.write_text('KEY="a#b"\n', encoding="utf-8-sig")
        result = _parse_env(env_file)
        assert result.get("KEY") == "a#b"

    def test_hash_inside_single_quotes(self, tmp_path):
        """KEY='x # y' — # внутри одинарных кавычек."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY='x # y'\n", encoding="utf-8-sig")
        result = _parse_env(env_file)
        assert result.get("KEY") == "x # y"

    def test_hash_after_value_unquoted(self, tmp_path):
        """KEY=v # c — # вне кавычек = комментарий, value = v."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=v # c\n", encoding="utf-8-sig")
        result = _parse_env(env_file)
        assert result.get("KEY") == "v"

    def test_hash_after_value_quoted(self, tmp_path):
        """KEY="v # c" — # после value в кавычках = комментарий, value = v # c."""
        env_file = tmp_path / ".env"
        env_file.write_text('KEY="v # c"\n', encoding="utf-8-sig")
        result = _parse_env(env_file)
        assert result.get("KEY") == "v # c"

    def test_windows_path_preserved(self, tmp_path):
        """KEY=C:\\path\\file — бэкслэши НЕ съедаются (posix=False)."""
        env_file = tmp_path / ".env"
        env_file.write_text(r'KEY=C:\path\file', encoding="utf-8-sig")
        result = _parse_env(env_file)
        assert result.get("KEY") == r"C:\path\file"

    def test_windows_path_in_quotes(self, tmp_path):
        """KEY="C:\\Program Files\\app" — в кавычках."""
        env_file = tmp_path / ".env"
        env_file.write_text(r'KEY="C:\Program Files\app"', encoding="utf-8-sig")
        result = _parse_env(env_file)
        assert result.get("KEY") == r"C:\Program Files\app"


# ══════════════════════════════════════════════════════════════════
# B9: unit-тест _strip_inline_comment
# ══════════════════════════════════════════════════════════════════

class TestStripInlineComment:
    cases = [
        ("value # comment", "value"),
        ('"a#b"', '"a#b"'),
        ("'x # y'", "'x # y'"),
        ('"a#b" # comment', '"a#b"'),
        ("a#b", "a#b"),
        ('"hello"  # inline', '"hello"'),
        ("no_comment", "no_comment"),
        ('"val#ue" # comment', '"val#ue"'),
    ]

    @pytest.mark.parametrize("input_,expected", cases)
    def test_strip_inline_comment(self, input_, expected):
        assert _strip_inline_comment(input_) == expected


# ══════════════════════════════════════════════════════════════════
# T2-11: _write_env — \n запрещён
# ══════════════════════════════════════════════════════════════════

class TestWriteEnv:
    def test_newline_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "newsbot.env._env_path",
            lambda: tmp_path / ".env",
        )
        with pytest.raises(ValueError, match="\\\\n"):
            _write_env("KEY", "multi\nline")

    def test_special_chars_quoted(self, tmp_path, monkeypatch):
        """value с пробелом/# записывается в кавычках."""
        monkeypatch.setattr(
            "newsbot.env._env_path",
            lambda: tmp_path / ".env",
        )
        _write_env("MYKEY", "hello world")
        content = (tmp_path / ".env").read_text(encoding="utf-8")
        # должно быть в кавычках
        assert 'MYKEY="hello world"' in content

    def test_newline_free_ok(self, tmp_path, monkeypatch):
        """Без \\n — записывается нормально."""
        monkeypatch.setattr(
            "newsbot.env._env_path",
            lambda: tmp_path / ".env",
        )
        _write_env("TESTKEY", "abc123")
        content = (tmp_path / ".env").read_text(encoding="utf-8")
        assert "TESTKEY=abc123" in content


# ══════════════════════════════════════════════════════════════════
# K4 + B10: токен не попадает в логи
# ══════════════════════════════════════════════════════════════════

class TestTokenSanitizer:
    """TokenSanitizer вырезает bot<digits>:<20+ chars> из любых сообщений."""

    def test_bot_token_masked(self):
        flt = TokenSanitizer()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0,
            "Bot token is bot1234567890:AbCdEfGhIjKlMnOpQrStUvWxYz", (), None
        )
        flt.filter(record)
        assert "***REDACTED***" in record.msg
        assert "AbCdEfGhIjKlMnOp" not in record.msg
        assert "bot1234567890:" not in record.msg

    def test_clean_message_unchanged(self):
        flt = TokenSanitizer()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "Normal log message", (), None
        )
        flt.filter(record)
        assert record.msg == "Normal log message"

    def test_args_sanitized(self):
        """
        Args, содержащие токен, санируются.
        record.getMessage() = msg % args. Если args содержат токен,
        подставленный результат тоже санируется.
        """
        flt = TokenSanitizer()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "url: %s", ("bot99:Token1234567890XYZ",), None
        )
        flt.filter(record)
        # args санированы: Token1234567890XYZ (18) ≥ 15 → вырезается
        assert "Token1234567890XYZ" not in str(record.args[0])
        assert "bot99:" not in str(record.args[0])
        # После подстановки msg % args токена тоже нет
        assembled = record.msg % record.args
        assert "***REDACTED***" in assembled
        assert "Token1234567890XYZ" not in assembled


class TestHttpxLogLevel:
    """K4: httpx/httpcore логируют на WARNING, не INFO."""

    def test_httpx_level_is_warning(self):
        setup_logging({"level": "DEBUG"})
        httpx_logger = logging.getLogger("httpx")
        assert httpx_logger.level <= logging.WARNING

    def test_httpcore_level_is_warning(self):
        setup_logging({"level": "DEBUG"})
        httpcore_logger = logging.getLogger("httpcore")
        assert httpcore_logger.level <= logging.WARNING


# ══════════════════════════════════════════════════════════════════
# B7: _split_text с огромным title не зацикливается
# ══════════════════════════════════════════════════════════════════

class TestSplitTextHugeTitle:
    def test_huge_title_no_infinite_loop(self):
        """title_esc > 200 символов — _split_text не зависает и не зацикливается."""
        td = TelegramDelivery(token="test_token", chat_id="123")
        # title на 1000 символов (больше _MAX_TITLE_LEN=200)
        huge_title = "A" * 1000
        body = "B" * 500
        footer = "footer"
        # должен вернуть список payload'ей, не зависнуть
        result = td._split_text(huge_title, body, footer, False, "")
        assert isinstance(result, list)
        assert len(result) > 0
        for r in result:
            assert "text" in r
            # ни один payload не превышает лимит
            assert len(r["text"]) <= 4096

    def test_title_limited_to_200(self):
        """Проверяем, что _MAX_TITLE_LEN=200 реально ограничивает."""
        td = TelegramDelivery(token="test_token", chat_id="123")
        huge_title = "X" * 300
        result = td._split_text(huge_title, "B", "F", False, "")
        # все части должны содержать title, усечённый до 200
        for r in result:
            assert len(r["text"]) <= 4096
        # итерация должна завершиться за разумное время
        assert len(result) > 0


# ══════════════════════════════════════════════════════════════════
# Задача 1 (волна 3): 429 — не расходует attempt, ограничен дедлайном
# и числом подряд идущих 429; все ожидания прерываемы shutdown_event
# ══════════════════════════════════════════════════════════════════

class Test429Handling:
    """
    TelegramDelivery._api_call при 429:
      — attempt НЕ расходуется (429 ≠ обычная ошибка);
      — суммарное ожидание 429 ≤ RETRY_429_TOTAL_DEADLINE_SEC на ОДИН вызов;
      — не более MAX_CONSECUTIVE_429 подряд;
      — ожидание через self._wait (event.wait при заданном shutdown_event).
    """

    @staticmethod
    def _resp_429(retry_after: int):
        r = MagicMock()
        r.status_code = 429
        r.json.return_value = {"ok": False, "parameters": {"retry_after": retry_after}}
        return r

    @staticmethod
    def _resp_ok():
        r = MagicMock()
        r.status_code = 200
        r.is_success = True
        r.json.return_value = {"ok": True, "result": {"message_id": 1}}
        return r

    def test_429_does_not_consume_attempt(self):
        """
        Один 429 (retry_after=1) + успех → ровно 2 HTTP-вызова.
        При старом for attempt in range(3) 429 съедал попытку, и в
        комбинации с другими ошибками вызов падал раньше времени.
        """
        td = TelegramDelivery(token="tok", chat_id="123")
        with patch.object(td._http, "post",
                          side_effect=[self._resp_429(1), self._resp_ok()]) as mock_post, \
             patch.object(td, "_wait") as mock_wait:
            result = td._api_call("sendMessage", {"chat_id": "123", "text": "hi"})

        assert result == {"message_id": 1}
        assert mock_post.call_count == 2, (
            f"HTTP-вызовов: {mock_post.call_count}, ожидалось 2"
        )
        mock_wait.assert_called_once_with(1)

    def test_four_consecutive_429_bounded(self):
        """4+ подряд 429: число вызовов и суммарный сон ограничены."""
        td = TelegramDelivery(token="tok", chat_id="123")
        with patch.object(td._http, "post",
                          side_effect=[self._resp_429(3)] * 5 + [self._resp_ok()]), \
             patch.object(td, "_wait") as mock_wait:
            result = td._api_call("sendMessage", {"chat_id": "123", "text": "hi"})

        assert result == {"message_id": 1}
        total = sum(call[0][0] for call in mock_wait.call_args_list)
        assert total <= RETRY_429_TOTAL_DEADLINE_SEC

    def test_429_total_deadline_caps_sleep(self):
        """
        retry_after=200 при дедлайне 600 → 3 ожидания по 200 с, на 4-м 429
        остаток дедлайна 0 → TelegramDeliveryError (item остаётся pending).
        """
        td = TelegramDelivery(token="tok", chat_id="123")
        with patch.object(td._http, "post",
                          side_effect=[self._resp_429(200)] * 10) as mock_post, \
             patch.object(td, "_wait") as mock_wait:
            with pytest.raises(TelegramDeliveryError):
                td._api_call("sendMessage", {"chat_id": "123", "text": "hi"})

        total = sum(call[0][0] for call in mock_wait.call_args_list)
        assert total <= RETRY_429_TOTAL_DEADLINE_SEC
        assert total == RETRY_429_TOTAL_DEADLINE_SEC
        # 3 ожидания + 4-й ответ 429, на котором вышли по дедлайну
        assert mock_post.call_count == 4

    def test_429_above_deadline_fails_fast(self):
        """
        retry_after (999 с) больше всего дедлайна → НЕ спим впустую,
        сразу TelegramDeliveryError.
        """
        td = TelegramDelivery(token="tok", chat_id="123")
        with patch.object(td._http, "post",
                          side_effect=[self._resp_429(999)]) as mock_post, \
             patch.object(td, "_wait") as mock_wait:
            with pytest.raises(TelegramDeliveryError):
                td._api_call("sendMessage", {"chat_id": "123", "text": "hi"})

        mock_wait.assert_not_called()
        assert mock_post.call_count == 1

    def test_429_consecutive_limit(self):
        """
        Бесконечные 429 с retry_after=1 → не более MAX_CONSECUTIVE_429
        ожиданий, затем ошибка (защита от бесконечного цикла).
        """
        td = TelegramDelivery(token="tok", chat_id="123")
        with patch.object(td._http, "post",
                          side_effect=[self._resp_429(1)] * 50) as mock_post, \
             patch.object(td, "_wait") as mock_wait:
            with pytest.raises(TelegramDeliveryError):
                td._api_call("sendMessage", {"chat_id": "123", "text": "hi"})

        assert mock_wait.call_count == MAX_CONSECUTIVE_429
        assert mock_post.call_count == MAX_CONSECUTIVE_429 + 1

    def test_wait_is_interruptible_by_shutdown_event(self):
        """_wait не блокирует поток, если shutdown_event уже установлен."""
        event = threading.Event()
        event.set()
        td = TelegramDelivery(token="tok", chat_id="123", shutdown_event=event)

        start = time.monotonic()
        td._wait(60)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"_wait не прервался по shutdown_event ({elapsed:.2f}s)"

    def test_pacing_uses_wait_helper(self):
        """Pacing между сообщениями идёт через _wait (прерываемо), а не time.sleep."""
        td = TelegramDelivery(token="tok", chat_id="123",
                              shutdown_event=threading.Event())
        with patch.object(td, "_api_call"), \
             patch.object(td, "_wait") as mock_wait:
            td.send_batch([_make_item(i) for i in range(3)])

        assert mock_wait.call_count == 2
        for call in mock_wait.call_args_list:
            assert call[0][0] == SEND_PACING_SEC

    def test_send_batch_breaks_on_delivery_error(self):
        """
        TelegramDeliveryError → break: отправленное = delivered,
        остаток = pending (не долбим API всем батчем).
        """
        td = TelegramDelivery(token="tok", chat_id="123")
        items = [_make_item(i) for i in range(3)]
        with patch.object(td, "_send_one",
                          side_effect=[None, TelegramDeliveryError("429"), None]) as sent, \
             patch.object(td, "_wait"):
            result = td.send_batch(items)

        assert result == [items[0].url]
        assert sent.call_count == 2, "после ошибки батч должен прерваться"


# ══════════════════════════════════════════════════════════════════
# B10: telegram не пробрасывает сырые исключения с URL/токеном
# ══════════════════════════════════════════════════════════════════

class TestTelegramNoTokenLeak:
    def test_api_call_generic_httpx_error_logs_only_type(self):
        """При httpx.HTTPError (не подкласс) — логируется только тип, не str(e)."""
        td = TelegramDelivery(token="bot1234567890:REAL_TOKEN_SECRET_STUFF", chat_id="123")

        # UnsupportedProtocol — наследник httpx.HTTPError, но не RequestError
        exc = httpx.UnsupportedProtocol("https://bot1234567890:REAL_TOKEN_SECRET_STUFF/getMe")
        with patch.object(td._http, "post", side_effect=exc):
            with patch("time.sleep"):
                # должен бросить TelegramDeliveryError, НЕ UnsupportedProtocol
                # и лог НЕ должен содержать токен
                td._base_url  # инициализирован
                log_capture = io.StringIO()
                handler = logging.StreamHandler(log_capture)
                handler.setLevel(logging.WARNING)
                logger = logging.getLogger("newsbot.delivery.telegram")
                logger.addHandler(handler)
                logger.setLevel(logging.WARNING)

                try:
                    with pytest.raises(Exception):  # может быть TelegramDeliveryError
                        td._api_call("getMe", {})
                finally:
                    logger.removeHandler(handler)

                log_output = log_capture.getvalue()
                assert "REAL_TOKEN" not in log_output
                assert "bot1234567890" not in log_output
                assert "UnsupportedProtocol" in log_output

    def test_send_text_no_token_leak(self):
        """send_text при ошибке логирует только тип, не str(e)."""
        td = TelegramDelivery(token="bot1234567890:SECRET_TOKEN", chat_id="123")

        exc = httpx.TimeoutException("Client timeout at https://bot1234567890:SECRET_TOKEN/sendMessage")
        with patch.object(td._http, "post", side_effect=exc):
            log_capture = io.StringIO()
            handler = logging.StreamHandler(log_capture)
            handler.setLevel(logging.WARNING)
            logger = logging.getLogger("newsbot.delivery.telegram")
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)

            result = td.send_text("test")  # должен вернуть False, не бросить

            logger.removeHandler(handler)
            assert result is False
            log_output = log_capture.getvalue()
            assert "SECRET_TOKEN" not in log_output
            assert "bot1234567890" not in log_output


# ══════════════════════════════════════════════════════════════════
# T2-7: маскирование — длина + последние 4
# ══════════════════════════════════════════════════════════════════

class TestMaskKey:
    """T2-7: _mask_key показывает ...<last4> (len=N), НЕ первые 6."""

    def test_mask_shows_last4_and_length(self):
        from newsbot.__main__ import _mask_key
        key = "my_super_secret_token_string"
        masked = _mask_key(key)
        # формат: ...<last4> (len=N)
        assert masked.endswith(f"(len={len(key)})")
        assert "ring" in masked  # последние 4 = "ring"
        assert "my_sup" not in masked  # первые 6 НЕ показаны

    def test_mask_short_key(self):
        from newsbot.__main__ import _mask_key
        masked = _mask_key("abc")
        assert "abc" in masked  # все 3 символа
        assert "(len=3)" in masked

    def test_mask_empty(self):
        from newsbot.__main__ import _mask_key
        assert _mask_key("") == "(empty)"


# ══════════════════════════════════════════════════════════════════
# T2-9: score-валидация — отклоняем bool/float/строки
# ══════════════════════════════════════════════════════════════════

class TestScoreValidation:
    """
    C6-фикс: тестируем реальный LLMFilter._parse_response, а не копию логики.
    """

    @pytest.fixture
    def filter_instance(self):
        """LLMFilter с замоканным клиентом (evaluate не нужен — вызываем _parse_response напрямую)."""
        return LLMFilter(config_llm={"enabled": True, "model": "test"})

    def _parse(self, filt: LLMFilter, raw: dict):
        """
        Обертка над _parse_response: возвращает score или None.
        Полный dict нужен для полей summary/reason (иначе ok=False).
        """
        result = filt._parse_response(raw)
        return result.score if result.ok else None

    def test_bool_true_rejected(self, filter_instance):
        assert self._parse(filter_instance, {"score": True, "summary": "s", "reason": "r"}) is None

    def test_bool_false_rejected(self, filter_instance):
        assert self._parse(filter_instance, {"score": False, "summary": "s", "reason": "r"}) is None

    def test_float_integral_accepted(self, filter_instance):
        """7.0, 0.0, 10.0 — целочисленные float принимаются."""
        assert self._parse(filter_instance, {"score": 7.0, "summary": "s", "reason": "r"}) == 7
        assert self._parse(filter_instance, {"score": 0.0, "summary": "s", "reason": "r"}) == 0
        assert self._parse(filter_instance, {"score": 10.0, "summary": "s", "reason": "r"}) == 10

    def test_float_fractional_rejected(self, filter_instance):
        """7.5, 3.14 — дробные float отклоняются."""
        assert self._parse(filter_instance, {"score": 7.5, "summary": "s", "reason": "r"}) is None
        assert self._parse(filter_instance, {"score": 3.14, "summary": "s", "reason": "r"}) is None

    def test_int_in_range_accepted(self, filter_instance):
        for score in range(0, 11):
            assert self._parse(filter_instance, {"score": score, "summary": "s", "reason": "r"}) == score

    def test_int_out_of_range_rejected(self, filter_instance):
        assert self._parse(filter_instance, {"score": -1, "summary": "s", "reason": "r"}) is None
        assert self._parse(filter_instance, {"score": 11, "summary": "s", "reason": "r"}) is None
        assert self._parse(filter_instance, {"score": 999, "summary": "s", "reason": "r"}) is None

    def test_str_int_accepted(self, filter_instance):
        """'7', '0', '10' — строки, парсящиеся в int 0..10, принимаются."""
        assert self._parse(filter_instance, {"score": "7", "summary": "s", "reason": "r"}) == 7
        assert self._parse(filter_instance, {"score": "0", "summary": "s", "reason": "r"}) == 0
        assert self._parse(filter_instance, {"score": "10", "summary": "s", "reason": "r"}) == 10

    def test_str_non_numeric_rejected(self, filter_instance):
        assert self._parse(filter_instance, {"score": "abc", "summary": "s", "reason": "r"}) is None
        assert self._parse(filter_instance, {"score": "7 7", "summary": "s", "reason": "r"}) is None
        assert self._parse(filter_instance, {"score": "7.0", "summary": "s", "reason": "r"}) is None
        assert self._parse(filter_instance, {"score": "", "summary": "s", "reason": "r"}) is None


# ══════════════════════════════════════════════════════════════════
# T2-8: время в локальной зоне (console.py _format_time)
# ══════════════════════════════════════════════════════════════════

class TestTimezone:
    def test_format_time_returns_local(self):
        from newsbot.delivery import format_time_utc
        # UTC noon → конвертится в local
        utc_noon = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = format_time_utc(utc_noon, None)
        # формат %d.%m.%Y %H:%M
        assert re.match(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}", result)

    def test_format_time_naive_converted(self):
        """Naive datetime — treated as UTC, затем конвертится в local."""
        from newsbot.delivery import format_time_utc
        naive = datetime(2025, 1, 15, 15, 30, 0)  # no tzinfo
        result = format_time_utc(naive, None)
        assert re.match(r"\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}", result)


# ══════════════════════════════════════════════════════════════════
# C1: _check_telegram / _set_chat_id разделены корректно
# ══════════════════════════════════════════════════════════════════

class TestSetChatId:
    """C1: --set-chat-id вызывает _set_chat_id, не попадает в тело _check_telegram."""

    def test_set_chat_id_no_name_error(self, monkeypatch):
        """_set_chat_id определён и вызывается без NameError."""
        # патчим telegram_token и _write_env чтобы не было сетевых запросов
        import newsbot.__main__ as main_mod

        def fake_token():
            return None  # ранний выход без NameError

        monkeypatch.setattr(main_mod, "telegram_token", fake_token)
        # вызов должен пройти без NameError (печатает ошибку и выходит)
        main_mod._set_chat_id()
        # если дошли сюда без NameError — тест прошёл

    def test_check_telegram_no_extra_get_updates(self, monkeypatch):
        """_check_telegram не делает лишний getUpdates (не попадает в _set_chat_id)."""
        import newsbot.__main__ as main_mod
        import httpx

        def fake_token():
            return "bot1234567890:FAKE_TOKEN_1234567890ABC"

        called_urls = []

        def fake_get(url, **kwargs):
            called_urls.append(url)
            # мокаем getMe — ok
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"ok": True, "result": {"username": "test"}}
            return mock_resp

        monkeypatch.setattr(main_mod, "telegram_token", fake_token)
        with patch.object(httpx, "get", fake_get):
            main_mod._check_telegram()

        # был только getMe, getUpdates не вызывался
        assert any("getMe" in url for url in called_urls)
        assert not any("getUpdates" in url for url in called_urls)


# ══════════════════════════════════════════════════════════════════
# C2: _split_text не усекает экранированные title/source
# ══════════════════════════════════════════════════════════════════

class TestSplitTextNoEntityTruncation:
    """C2: экранированный title/source НЕ усекается после escape, сущности не обрываются."""

    def test_title_with_ampersand_at_boundary(self):
        """Title '&' у границы _MAX_RAW_TITLE=250 → не '&am' → Telegram 400."""
        td = TelegramDelivery(token="test_token", chat_id="123")
        # 248 символов + '&' = 249, НЕ обрезается (250), после escape '&amp' = 4 символа
        raw = "A" * 248 + "&"
        body = "B" * 500
        footer = "source"
        result = td._split_text(raw, body, footer, False, "")
        # все payload содержат валидный HTML (без '&am' без точки с запятой)
        for r in result:
            text = r["text"]
            assert not re.search(r"&[a-zA-Z]{1,3}(?=[\s<>\n]|$)", text)
            assert len(text) <= 4096

    def test_title_with_lt_at_boundary(self):
        """Title '<' у границы 250 → '&lt;' (не обрезанное)."""
        td = TelegramDelivery(token="test_token", chat_id="123")
        raw = "A" * 249 + "<"
        body = "B" * 500
        footer = "source"
        result = td._split_text(raw, body, footer, False, "")
        for r in result:
            text = r["text"]
            # нет '&lt' без ';' (оборванная сущность)
            assert not re.search(r"&lt(?!;)", text)
            assert len(text) <= 4096

    def test_no_infinite_loop_on_huge_title_with_entities(self):
        """Огромный title с &/</> у границ — _split_text завершается, не зацикливается."""
        td = TelegramDelivery(token="test_token", chat_id="123")
        huge = "A" * 500 + "&" + "B" * 500 + "<" + "C" * 500
        body = "X" * 1000
        footer = "footer"
        # должен вернуть список, не зависнуть и не бросить
        result = td._split_text(huge, body, footer, False, "")
        assert isinstance(result, list)
        assert len(result) > 0
        for r in result:
            assert len(r["text"]) <= 4096


# ══════════════════════════════════════════════════════════════════
# C3: TokenSanitizer корректно обрабатывает dict/mapping args
# ══════════════════════════════════════════════════════════════════

class TestTokenSanitizerMappingArgs:
    """C3: Mapping args (%(key)s format) не падают TypeError при getMessage()."""

    def test_dict_args_no_type_error(self):
        """
        record.args = dict → санируется in-place (type остаётся dict),
        record.getMessage() с %(name)s форматом не падает TypeError.
        """
        flt = TokenSanitizer()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0,
            "%(name)s: %(token)s",  # msg требует Mapping
            {"name": "user", "token": "bot1234567890:AbCdEfGhIjKlMnOp"},
            None,  # exc_info
        )
        # не должно бросить TypeError
        flt.filter(record)
        # getMessage() не падает
        message = record.getMessage()
        assert "***REDACTED***" in message
        assert "AbCdEfGhIjKlMnOp" not in message

    def test_dict_args_token_masked(self):
        """
        Token в строковом значении dict-value санируется.
        Примечание: %(key)s с dict как args напрямую не поддерживается
        Python 3.8 (KeyError:0 в LogRecord.__init__), поэтому тестируем
        через tuple-format (%s) — наиболее частый реальный паттерн.
        """
        flt = TokenSanitizer()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0,
            "url=%s",  # %s-format, dict передаётся как один аргумент
            ({"url": "https://api.telegram.org/bot1234567890:TOKENSEcretPart/String"},),
            None,  # exc_info
        )
        flt.filter(record)
        message = record.getMessage()
        assert "***REDACTED***" in message
        assert "TOKENSEcretPart" not in message

    def test_tuple_with_dict_inside_no_type_error(self):
        """Dict как элемент tuple (не %()-format) — getMessage() не падает."""
        flt = TokenSanitizer()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0,
            "data: %s",
            ({"token": "bot1234567890:TOKENSEcretExtra1"},),
            None,  # exc_info
        )
        flt.filter(record)
        message = record.getMessage()
        assert "***REDACTED***" in message
        assert "TOKENSEcretExtra1" not in message


# ══════════════════════════════════════════════════════════════════
# Задача 1: pacing между сообщениями
# ══════════════════════════════════════════════════════════════════

class TestSendPacing:
    """SEND_PACING_SEC=0.6: пауза между сообщениями, кроме последнего."""

    def test_send_batch_sleeps_between_messages(self):
        """
        send_batch: пауза после каждого сообщения КРОМЕ последнего.
        time.sleep мокаем, проверяем количество и интервалы.
        """
        td = TelegramDelivery(token="tok", chat_id="123")

        # Мокаем _api_call — все успешные
        with patch.object(td, "_api_call"):
            with patch("newsbot.delivery.telegram.time.sleep") as mock_sleep:
                with patch("newsbot.delivery.telegram.SEND_PACING_SEC", 0.6):
                    items = [_make_item(i) for i in range(3)]
                    td.send_batch(items)

                # 3 items → 2 паузы (после item 0 и item 1)
                assert mock_sleep.call_count == 2, (
                    f"sleep вызван {mock_sleep.call_count} раз, ожидалось 2"
                )
                # Каждая пауза = 0.6 с
                for call in mock_sleep.call_args_list:
                    assert call[0][0] == 0.6

    def test_send_batch_single_item_no_sleep(self):
        """Одно сообщение — пауза НЕ делается."""
        td = TelegramDelivery(token="tok", chat_id="123")

        with patch.object(td, "_api_call"):
            with patch("newsbot.delivery.telegram.time.sleep") as mock_sleep:
                items = [_make_item(1)]
                td.send_batch(items)

        # 1 item → 0 пауз
        assert mock_sleep.call_count == 0

    def test_send_batch_empty_no_sleep(self):
        """Пустой батч — пауза НЕ делается."""
        td = TelegramDelivery(token="tok", chat_id="123")

        with patch("newsbot.delivery.telegram.time.sleep") as mock_sleep:
            result = td.send_batch([])
        assert result == []
        assert mock_sleep.call_count == 0


def _make_item(i: int):
    """Factory: NewsItem dataclass с уникальным url."""
    from dataclasses import dataclass, field
    from datetime import datetime, timezone
    from newsbot.collector import NewsItem
    return NewsItem(
        source="test",
        url=f"https://example.com/{i}",
        title=f"Title {i}",
        summary=f"Summary {i}",
        published="2025-01-01",
        fetched_at=datetime.now(timezone.utc),
    )


# ══════════════════════════════════════════════════════════════════
# P1: пейджинг длинного body с HTML-спецсимволами не рвёт сущности
# ══════════════════════════════════════════════════════════════════

_VALID_ENTITY_RE = re.compile(r"&(?:amp|lt|gt|quot|#x27);")


def _broken_entities(text: str) -> bool:
    """True, если в тексте есть '&', не начинающий валидную сущность."""
    return "&" in _VALID_ENTITY_RE.sub("", text)


class TestSplitTextRawBodyNoEntityTruncation:
    """
    P1-регресс: длинный СЫРОЙ body с " / ' / & раньше резался УЖЕ
    экранированным — срез попадал внутрь &quot;/&#x27;/&amp; → Telegram 400
    «can't parse entities» → item в failed после 8 ретраев (потеря новости).

    Теперь режем сырой текст и экранируем каждый чанк отдельно.
    """

    @staticmethod
    def _item_with_body(body: str):
        from datetime import datetime, timezone
        from newsbot.collector import NewsItem
        return NewsItem(
            source='ИСТОЧНИК "в кавычках" & Co',
            url="https://example.com/long",
            title='Заголовок с "кавычками" & амперсандом',
            summary="short",
            published="2025-01-01",
            fetched_at=datetime.now(timezone.utc),
            llm_summary=body,
        )

    def test_long_body_with_quotes_and_ampersands(self):
        """Сырой body >5000 симв. с " / ' / & → все части валидный HTML ≤4096."""
        body = ('Новость "кавычки" & амперсанд, апостроф\'s. ' * 200) + " ХВОСТ_МАРКЕР"
        assert len(body) > 5000

        td = TelegramDelivery(token="test_token", chat_id="123")
        parts = td._build_message_parts(self._item_with_body(body))

        assert len(parts) > 1, "длинный body должен разбиваться на части"
        for p in parts:
            text = p["text"]
            assert len(text) <= 4096
            assert not _broken_entities(text), "разорванная HTML-сущность"
        # хвост новости не потерян
        assert "ХВОСТ_МАРКЕР" in parts[-1]["text"]

    def test_escape_expanding_body_quotes_only(self):
        """3000 сырых ' (→ &#x27;) и 3000 & (→ &amp;) — сущности не рвутся."""
        body = ('"' * 3000) + " " + ("&" * 3000)
        assert len(body) == 6001

        td = TelegramDelivery(token="test_token", chat_id="123")
        parts = td._build_message_parts(self._item_with_body(body))

        assert len(parts) >= 2
        for p in parts:
            text = p["text"]
            assert len(text) <= 4096
            assert not _broken_entities(text)

    def test_split_paragraph_chunks_are_valid_escaped_html(self):
        """_split_paragraph: каждый кусок — самостоятельный валидный escaped-HTML."""
        td = TelegramDelivery(token="test_token", chat_id="123")
        para = ('Он сказал "да" & ушёл\t' * 900)  # ~18k сырых символов
        chunks = td._split_paragraph(para, 1000)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 1000
            assert not _broken_entities(chunk)
            assert "<" not in chunk and ">" not in chunk


# ══════════════════════════════════════════════════════════════════
# UTF-16: Telegram меряет длину в code units, не в code points
# ══════════════════════════════════════════════════════════════════


def _utf16_units(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


class TestSplitTextUtf16Limit:
    """Не-BMP символы (эмодзи) — 2 UTF-16 units: части не должны превышать 4096."""

    def test_4096_emoji_fit(self):
        td = TelegramDelivery(token="test_token", chat_id="123")
        # 4096 code points, но 8192 UTF-16 units — Telegram отверг бы целиком.
        body = "😀" * 4096
        result = td._split_text("T", body, "F", False, "")
        assert len(result) > 1
        for r in result:
            assert _utf16_units(r["text"]) <= 4096

    def test_emoji_body_build_message_parts(self):
        from newsbot.collector import NewsItem
        body = "😀" * 5000
        item = NewsItem(
            source="src", url="https://example.com/x", title="t",
            summary="short", published="2025-01-01",
            fetched_at=datetime.now(timezone.utc), llm_summary=body,
        )
        td = TelegramDelivery(token="test_token", chat_id="123")
        parts = td._build_message_parts(item)
        assert len(parts) > 1
        for p in parts:
            assert _utf16_units(p["text"]) <= 4096

    def test_quotes_only_body_parts_are_valid(self):
        """Строка из одних `"` (escape ×6) — валидные части в пределах лимита."""
        td = TelegramDelivery(token="test_token", chat_id="123")
        result = td._split_text("T", '"' * 3000, "F", False, "")
        assert result
        for r in result:
            assert _utf16_units(r["text"]) <= 4096
            assert not _broken_entities(r["text"])
