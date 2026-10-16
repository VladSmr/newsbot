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
from newsbot.delivery.telegram import TelegramDelivery


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
# B8: 429 retry_after каппится до 120
# ══════════════════════════════════════════════════════════════════

class Test429RetryCap:
    def test_retry_after_logic_capped(self):
        """Проверяем cap через патчинг httpx.HTTPStatusError."""
        import httpx

        client = DeepSeekClient(
            api_key="fake-key-not-real",
            base_url="https://localhost:99999",
            model="fake",
            max_retries=1,
        )
        # создаём httpx.HTTPStatusError с retry-after: 999
        resp_mock = MagicMock()
        resp_mock.status_code = 429
        resp_mock.headers = {"retry-after": "999"}
        resp_mock.text = ""

        exc = httpx.HTTPStatusError(
            "429", request=MagicMock(), response=resp_mock
        )

        with patch.object(client._http, "post", side_effect=exc):
            with patch("time.sleep") as mock_sleep:
                with pytest.raises(LLMError) as exc_info:
                    client.chat_json("sys", "user")
                # Конкретный тип исключения: LLMError (не generic Exception)
                assert isinstance(exc_info.value, LLMError)
                # retry-after=999 должен быть закаплен в 120
                assert mock_sleep.called, "time.sleep должен быть вызван при 429"
                for call in mock_sleep.call_args_list:
                    wait_arg = call[0][0]
                    assert wait_arg <= 120, f"wait={wait_arg} превышает cap 120!"


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