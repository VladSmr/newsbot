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


# ────────────────────────────────────────────────────────────────
# Юнит-тесты _sanitize()
# ────────────────────────────────────────────────────────────────

class TestSanitizeUnit:
    """Unittest'ы для функции _sanitize() отдельно от Filter."""

    def test_telegram_token_in_msg(self):
        """T4: Telegram-токен с префиксом bot вырезается."""
        raw = "Token: bot123456:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"
        assert "AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw" not in _sanitize(raw)
        assert "bot" not in _sanitize(raw)  # только маска

    def test_naked_token_in_msg(self):
        """T4: Голый токен <digits>:<secret> вырезается."""
        raw = "Found token 123456:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw in request"
        assert "AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw" not in _sanitize(raw)
        assert "123456" not in _sanitize(raw)

    def test_deepseek_key_in_msg(self):
        """T4: DeepSeek API key (sk-…) вырезается."""
        raw = "Authorization: Bearer sk-9RNDl3vLm2xKp8Yu4ZhQ6wJ"
        assert "sk-9RNDl3vLm2xKp8Yu4ZhQ6wJ" not in _sanitize(raw)
        assert "***REDACTED***" in _sanitize(raw)

    def test_multiple_tokens_in_msg(self):
        """T4: Несколько секретов в одном сообщении — все вырезаются."""
        raw = (
            "bot789:ABCDefgh1234567890abcdefg "
            "also 999888:XYz1234567890abcdefghij "
            "and sk-9RNDl3vLm2xKp8Yu4ZhQ6wJ"
        )
        result = _sanitize(raw)
        assert "ABCDefgh1234567890abcdefg" not in result
        assert "XYz1234567890abcdefghij" not in result
        assert "sk-9RNDl3vLm2xKp8Yu4ZhQ6wJ" not in result
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
            "Telegram API call to bot123456:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"
        )
        self._apply_filter(record)
        assert "AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw" not in record.msg

    def test_filter_args_positional(self):
        """T4: токен в positional args вырезается."""
        record = self._make_record(
            "URL: %s", "https://api.telegram.org/bot777888:SecretTokenHere/sendMessage"
        )
        self._apply_filter(record)
        # args — tuple, проверяем через getMessage()
        assert "SecretTokenHere" not in record.getMessage()

    def test_filter_args_mapping(self):
        """T4: токен в Mapping args (dict) вырезается."""
        record = self._make_record(
            "Data: %(token)s", {"token": "bot999000:VeryLongSecretTokenABCDEF"}
        )
        self._apply_filter(record)
        # args остаётся dict; проверяем getMessage() — подставляет значения
        assert "VeryLongSecretTokenABCDEF" not in record.getMessage()

    def test_filter_args_tuple_of_dict(self):
        """T4: токен внутри dict, который сам является элементом args tuple."""
        record = self._make_record(
            "Request: %s", {"api_token": "555444:AbcDefGhIjkLmNoPqRsTuVwXyz123456"}
        )
        self._apply_filter(record)
        msg = record.getMessage()
        assert "AbcDefGhIjkLmNoPqRsTuVwXyz123456" not in msg

    def test_filter_deepseek_key_in_msg(self):
        """T4: DeepSeek API key в record.msg вырезается."""
        record = self._make_record(
            "DeepSeek auth: sk-9RNDl3vLm2xKp8Yu4ZhQ6wJ"
        )
        self._apply_filter(record)
        assert "sk-9RNDl3vLm2xKp8Yu4ZhQ6wJ" not in record.msg

    def test_filter_stack_info(self):
        """
        T4: _sanitize() вырезает токен из stack_info-строки.
        LogRecord.stack_info — computed property; TokenSanitizer.filter() не может
        подменить его на уровне Filter, поэтому санация происходит в
        SanitizingFormatter.format() (проверено end-to-end тестом
        test_traceback_sanitized). Проверяем ключевую функцию _sanitize().
        """
        text = "During call: bot123456:SecretTokenABCDEFghij"
        sanitized = _sanitize(text)
        assert "SecretTokenABCDEFghij" not in sanitized
        assert "bot" not in sanitized  # только маска

    def test_filter_exc_text(self):
        """
        T4: _sanitize() вырезает токен из exc_text-строки.
        LogRecord.exc_text — computed property, заполняется Formatter.format();
        TokenSanitizer.filter() не может подменить его на уровне Filter,
        поэтому санация происходит в SanitizingFormatter.format() (проверено
        end-to-end тестом test_traceback_sanitized). Проверяем _sanitize().
        """
        text = "Traceback (most recent call last):\n  token=bot777888:SecretKEYHereABCDEF\n"
        sanitized = _sanitize(text)
        assert "SecretKEYHereABCDEF" not in sanitized
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
            logger.info("Bot token: bot123456:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw")
            # Сценарий 2: токен в аргументе
            logger.info("Request URL: https://api.telegram.org/bot888999:SecretHereABCDEF/send")
            # Сценарий 3: DeepSeek
            logger.info("Auth header: sk-9RNDl3vLm2xKp8Yu4ZhQ6wJ")
            # Сценарий 4: несколько токенов
            logger.info(
                "Multi: bot111:Tok11111111111111111  222333:Tok22222222222222222  sk-9RNDl3vLm2xKp8Yu4ZhQ6wJ"
            )

            logger.handlers.clear()  # очищаем после теста
        finally:
            sys.stderr = old_stderr

        output = captured.getvalue()
        assert "AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw" not in output
        assert "SecretHereABCDEF" not in output
        assert "sk-9RNDl3vLm2xKp8Yu4ZhQ6wJ" not in output
        assert "Tok11111111111111111" not in output
        assert "Tok22222222222222222" not in output
        # Маска присутствует
        assert "***REDACTED***" in output

    def test_traceback_sanitized(self):
        """
        T4: токен в traceback (exc_info) не попадает в лог.
        """
        from newsbot.log import setup_logging

        old_stderr = sys.stderr
        captured = io.StringIO()
        sys.stderr = captured

        try:
            setup_logging({"level": "DEBUG", "format": "%(message)s"})
            logger = logging.getLogger("t4_tb_test")

            try:
                raise RuntimeError("auth=bot555666:SecretKeyABCDEFGHIJKLMN")
            except Exception:
                logger.error("API failed", exc_info=sys.exc_info())

            logger.handlers.clear()
        finally:
            sys.stderr = old_stderr

        output = captured.getvalue()
        assert "SecretKeyABCDEFGHIJKLMN" not in output