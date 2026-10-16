"""
tests/test_log_sanitizer.py — T4-фикс
Тесты для TokenSanitizer и _sanitize() в log.py.
Проверяют вырезание Telegram-токенов, DeepSeek API key и голых токенов
из msg, args и traceback (exc_info).
"""
import logging
import sys
import io

import pytest

from newsbot.log import TokenSanitizer, _sanitize

# ─── Паттерны-заглушки (35 символов, ≥15, гарантированно матчатся регекспом) ───
_TG_SECRET = "A" * 35
_TG_TOKEN = "bot" + "123456" + ":" + _TG_SECRET     # матчит re: bot\d+:[A-Za-z0-9_-]{15,}
_BARE_TOKEN = "123456" + ":" + _TG_SECRET            # матчит re: \d+:[A-Za-z0-9_-]{15,}
_DS_KEY = "sk-" + "a" * 30                           # матчит re: sk-[A-Za-z0-9]{16,}


# ────────────────────────────────────────────────────────────────
# Юнит-тесты _sanitize()
# ────────────────────────────────────────────────────────────────

class TestSanitizeUnit:
    """Unittest'ы для функции _sanitize() отдельно от Filter."""

    def test_telegram_token_in_msg(self):
        """T4: Telegram-токен с префиксом bot вырезается."""
        raw = "Token: " + _TG_TOKEN
        sanitized = _sanitize(raw)
        assert _TG_SECRET not in sanitized
        assert "bot" not in sanitized  # только маска

    def test_naked_token_in_msg(self):
        """T4: Голый токен <digits>:<secret> вырезается."""
        raw = "Found token " + _BARE_TOKEN + " in request"
        sanitized = _sanitize(raw)
        assert _TG_SECRET not in sanitized
        assert "123456" not in sanitized

    def test_deepseek_key_in_msg(self):
        """T4: DeepSeek API key (sk-…) вырезается."""
        raw = "Authorization: Bearer " + _DS_KEY
        sanitized = _sanitize(raw)
        assert _DS_KEY not in sanitized
        assert "***REDACTED***" in sanitized

    def test_multiple_tokens_in_msg(self):
        """T4: Несколько секретов в одном сообщении — все вырезаются."""
        raw = (
            "bot789:ABCDefgh1234567890abcdefg "
            "also 999888:XYz1234567890abcdefghij "
            "and " + _DS_KEY
        )
        result = _sanitize(raw)
        assert "ABCDefgh1234567890abcdefg" not in result
        assert "XYz1234567890abcdefghij" not in result
        assert _DS_KEY not in result
        # маска присутствует ≥ 3 раз
        assert result.count("***REDACTED***") >= 3

    def test_no_token_unchanged(self):
        """Санированный текст без токена не меняется."""
        clean = "Normal log message without secrets"
        assert _sanitize(clean) == clean

    def test_non_string_passthrough(self):
        """Non-str не падают, а возвращаются как есть."""
        assert _sanitize(123) == 123
        assert _sanitize(None) is None

    def test_short_token_not_matched(self):
        """Короткий secret (<15 символов) не matched (ложно-положительные)."""
        raw = "token=abc12345:short"
        assert _sanitize(raw) == raw  # не найдено, не меняется


# ────────────────────────────────────────────────────────────────
# Интеграционные тесты TokenSanitizer как logging.Filter
# ────────────────────────────────────────────────────────────────

class TestTokenSanitizerFilter:
    """TokenSanitizer.filter() — проверяем санацию полей LogRecord."""

    def _make_record(self, msg, *args, exc_info=None, stack_info=None):
        """Вспомогательник: создаёт LogRecord с заданными полями."""
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname=__file__, lineno=0,
            msg=msg, args=args,
            exc_info=exc_info, stack_info=stack_info,
        )
        return record

    def _apply_filter(self, record):
        sf = TokenSanitizer()
        return sf.filter(record)

    def test_filter_msg_token(self):
        """T4: токен в record.msg вырезается."""
        record = self._make_record(
            "Telegram API call to " + _TG_TOKEN
        )
        self._apply_filter(record)
        assert _TG_SECRET not in record.msg

    def test_filter_args_positional(self):
        """T4: токен в positional args вырезается."""
        _secret = "SecretTokenHere"
        _token = "bot777888:" + _secret
        record = self._make_record(
            "URL: %s", "https://api.telegram.org/" + _token + "/sendMessage"
        )
        self._apply_filter(record)
        # args — tuple, проверяем через getMessage()
        assert _secret not in record.getMessage()

    def test_filter_args_mapping(self):
        """T4: токен в Mapping args (dict) вырезается."""
        _secret = "VeryLongSecretTokenABCDEF"
        _token = "bot999000:" + _secret
        record = self._make_record(
            "Data: %(token)s", {"token": _token}
        )
        self._apply_filter(record)
        # args остаётся dict; проверяем getMessage() — подставляет значения
        assert _secret not in record.getMessage()

    def test_filter_args_tuple_of_dict(self):
        """T4: токен внутри dict, который сам является элементом args tuple."""
        _secret = "AbcDefGhIjkLmNoPqRsTuVwXyz123456"
        _token = "555444:" + _secret
        record = self._make_record(
            "Request: %s", {"api_token": _token}
        )
        self._apply_filter(record)
        msg = record.getMessage()
        assert _secret not in msg

    def test_filter_deepseek_key_in_msg(self):
        """T4: DeepSeek API key в record.msg вырезается."""
        record = self._make_record(
            "DeepSeek auth: " + _DS_KEY
        )
        self._apply_filter(record)
        assert _DS_KEY not in record.msg

    def test_filter_stack_info(self):
        """
        T4: _sanitize() вырезает токен из stack_info-строки.
        LogRecord.stack_info — computed property; TokenSanitizer.filter() не может
        подменить его на уровне Filter, поэтому санация происходит в
        SanitizingFormatter.format() (проверено end-to-end тестом
        test_traceback_sanitized). Проверяем ключевую функцию _sanitize().
        """
        text = "During call: " + _TG_TOKEN
        sanitized = _sanitize(text)
        assert _TG_SECRET not in sanitized
        assert "bot" not in sanitized  # только маска

    def test_filter_exc_text(self):
        """
        T4: _sanitize() вырезает токен из exc_text-строки.
        LogRecord.exc_text — computed property, заполняется Formatter.format();
        TokenSanitizer.filter() не может подменить его на уровне Filter,
        поэтому санация происходит в SanitizingFormatter.format() (проверено
        end-to-end тестом test_traceback_sanitized). Проверяем _sanitize().
        """
        _secret = "SecretKEYHereABCDEF"
        _token = "bot777888:" + _secret
        text = "Traceback (most recent call last):\n  token=" + _token + "\n"
        sanitized = _sanitize(text)
        assert _secret not in sanitized
        assert "bot" not in sanitized  # только маска


# ────────────────────────────────────────────────────────────────
# Интеграционный тест через end-to-end логгирование
# ────────────────────────────────────────────────────────────────

class TestSanitizerEndToEnd:
    """Проверка через реальный Handler + Formatter — полный конвейер."""

    def test_no_secret_in_output(self):
        """
        T4: токен Telegram и DeepSeek ключ не попадают в итоговый
        отформатированный вывод.
        """
        from newsbot.log import setup_logging

        # Перехватываем stderr
        old_stderr = sys.stderr
        captured = io.StringIO()
        sys.stderr = captured

        try:
            # Чистая конфигурация логгера
            setup_logging({"level": "DEBUG", "format": "%(message)s"})
            logger = logging.getLogger("t4_test")

            # Сценарий 1: токен в msg
            logger.info("Bot token: " + _TG_TOKEN)
            # Сценарий 2: токен в аргументе
            _secret2 = "SecretHereABCDEF"
            logger.info("Request URL: https://api.telegram.org/bot888999:" + _secret2 + "/send")
            # Сценарий 3: DeepSeek
            logger.info("Auth header: " + _DS_KEY)
            # Сценарий 4: несколько токенов
            logger.info(
                "Multi: bot111:Tok11111111111111111  222333:Tok22222222222222222  " + _DS_KEY
            )

            logger.handlers.clear()  # очищаем после теста
        finally:
            sys.stderr = old_stderr

        output = captured.getvalue()
        assert _TG_SECRET not in output
        assert _secret2 not in output
        assert _DS_KEY not in output
        assert "Tok11111111111111111" not in output
        assert "Tok22222222222222222" not in output
        # Маска присутствует
        assert "***REDACTED***" in output

    def test_traceback_sanitized(self):
        """
        T4: токен в traceback (exc_info) не попадает в лог.
        """
        from newsbot.log import setup_logging

        _secret = "SecretKeyABCDEFGHIJKLMN"
        _token = "bot555666:" + _secret

        old_stderr = sys.stderr
        captured = io.StringIO()
        sys.stderr = captured

        try:
            setup_logging({"level": "DEBUG", "format": "%(message)s"})
            logger = logging.getLogger("t4_tb_test")

            try:
                raise RuntimeError("auth=" + _token)
            except Exception:
                logger.error("API failed", exc_info=sys.exc_info())

            logger.handlers.clear()
        finally:
            sys.stderr = old_stderr

        output = captured.getvalue()
        assert _secret not in output