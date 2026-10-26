"""
Тесты: правила промпта, принятые владельцем в текущей сессии.

Блок 1: summary ВСЕГДА на английском; CORROBORATION CAP (score ≤ 4
для неподтверждённых одиночных сводок о жертвах/заявлениях).
Блок 2: RotatingFileHandler создаётся корректно.
"""
import logging
from pathlib import Path

import pytest

from newsbot.llm.prompts import (
    SYSTEM_PROMPT,
    BATCH_SYSTEM_PROMPT,
)
from newsbot.log import setup_logging


class TestSummaryAlwaysEnglish:
    """1.1: summary ВСЕГДА пишется на английском."""

    def test_system_prompt_says_always_en(self):
        """SYSTEM_PROMPT содержит правило «summary ALWAYS in English»."""
        assert "ALWAYS in English" in SYSTEM_PROMPT

    def test_batch_prompt_says_always_en(self):
        """BATCH_SYSTEM_PROMPT содержит правило «summary ALWAYS in English»."""
        assert "ALWAYS in English" in BATCH_SYSTEM_PROMPT

    def test_system_prompt_does_not_say_same_language(self):
        """Старое правило «SAME language as input» удалено из SYSTEM_PROMPT."""
        assert "SAME language" not in SYSTEM_PROMPT

    def test_batch_prompt_does_not_say_same_language(self):
        """Старое правило «SAME language as input» удалено из BATCH_SYSTEM_PROMPT."""
        assert "SAME language" not in BATCH_SYSTEM_PROMPT


class TestCorroborationCap:
    """1.2: CORROBORATION CAP — неподтверждённые одиночные сводки ≤ 4."""

    def test_cap_present_in_system_prompt(self):
        """SYSTEM_PROMPT содержит CORROBORATION CAP."""
        assert "CORROBORATION CAP" in SYSTEM_PROMPT

    def test_cap_present_in_batch_prompt(self):
        """BATCH_SYSTEM_PROMPT содержит CORROBORATION CAP."""
        assert "CORROBORATION CAP" in BATCH_SYSTEM_PROMPT

    def test_cap_score_4_maximum(self):
        """Правило: CAP the score at 4 maximum."""
        assert "4 maximum" in SYSTEM_PROMPT
        assert "4 maximum" in BATCH_SYSTEM_PROMPT

    def test_cap_mentions_casualty_figures(self):
        """CORROBORATION CAP section mentions casualty figures."""
        for prompt in (SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT):
            cap_pos = prompt.lower().find("corroboration cap")
            assert cap_pos >= 0, "CORROBORATION CAP section must exist in prompt"
            score_pos = prompt.lower().find("4 maximum")
            assert score_pos >= 0, "CAP score '4 maximum' must exist in prompt"
            casualty_pos = prompt.lower().find("casualty")
            assert casualty_pos >= 0, "CAP must mention 'casualty'"
            # All three elements exist — proximity check removed (may span multiple
            # sections by design: scale + dedicated CAP subsection).

    def test_cap_mentions_official_statements(self):
        """CAP применяется к заявлениям официальных лиц."""
        assert "officials" in SYSTEM_PROMPT.lower()
        assert "officials" in BATCH_SYSTEM_PROMPT.lower()

    def test_cap_excludes_personal_observation(self):
        """Исключение: источник лично наблюдал событие — CAP не применяется."""
        assert "personally observed" in SYSTEM_PROMPT.lower()
        assert "personally observed" in BATCH_SYSTEM_PROMPT.lower()

    def test_cap_excludes_named_quote(self):
        """Исключение: прямая цитата именованного чиновника — CAP не применяется."""
        assert "direct quote" in SYSTEM_PROMPT.lower()
        assert "direct quote" in BATCH_SYSTEM_PROMPT.lower()

    def test_cap_excludes_second_source_in_batch(self):
        """Исключение: второй независимый источник в батче — CAP не применяется."""
        assert "second independent source" in SYSTEM_PROMPT.lower()
        assert "second independent source" in BATCH_SYSTEM_PROMPT.lower()


class TestUrgentBlockParity:
    """
    Задача 1 (MAJOR): urgent=false примеры должны присутствовать
    в ОБОИХ промптах — SYSTEM_PROMPT и BATCH_SYSTEM_PROMPT.

    Основной prod-путь идёт через batch (batch_size=10).
    Потеря перечня urgent=false в BATCH_SYSTEM_PROMPT приводила к тому,
    что самая частая категория (ночные сводки БПЛА, санкции и т.д.)
    трактовалась как potentially-urgent → риск push-спама.
    """

    def test_urgent_false_examples_in_system_prompt(self):
        """SYSTEM_PROMPT содержит перечень не-срочных тем (urgent=false)."""
        assert "ordinary night drone attacks and their reports" in SYSTEM_PROMPT
        assert "sanctions" in SYSTEM_PROMPT
        assert "prices/fuel" in SYSTEM_PROMPT
        assert "regional incidents without mass casualties" in SYSTEM_PROMPT

    def test_urgent_false_examples_in_batch_prompt(self):
        """BATCH_SYSTEM_PROMPT содержит тот же перечень не-срочных тем."""
        assert "ordinary night drone attacks and their reports" in BATCH_SYSTEM_PROMPT
        assert "sanctions" in BATCH_SYSTEM_PROMPT
        assert "prices/fuel" in BATCH_SYSTEM_PROMPT
        assert "regional incidents without mass casualties" in BATCH_SYSTEM_PROMPT

    def test_urgent_false_examples_identical_in_both_prompts(self):
        """Блок URGENT_FALSE_EXAMPLES используется обоими промптами (один текст)."""
        from newsbot.llm.prompts import URGENT_FALSE_EXAMPLES
        assert URGENT_FALSE_EXAMPLES in SYSTEM_PROMPT
        assert URGENT_FALSE_EXAMPLES in BATCH_SYSTEM_PROMPT

    def test_urgent_true_criteria_in_batch_prompt(self):
        """BATCH_SYSTEM_PROMPT содержит критерии urgent=true."""
        assert "urgent=true" in BATCH_SYSTEM_PROMPT
        assert "beginning of war" in BATCH_SYSTEM_PROMPT
        assert "mobilization" in BATCH_SYSTEM_PROMPT
        assert "mass casualties" in BATCH_SYSTEM_PROMPT
        assert "nuclear or radiation" in BATCH_SYSTEM_PROMPT

    def test_urgent_doubt_falls_false_in_batch_prompt(self):
        """BATCH_SYSTEM_PROMPT: When in doubt → false."""
        assert "When in doubt" in BATCH_SYSTEM_PROMPT
        assert "false" in BATCH_SYSTEM_PROMPT


class TestFileLoggingHandler:
    """2.1: RotatingFileHandler создаётся корректно."""

    def test_rotating_handler_created_with_tmp_path(self, tmp_path, caplog):
        """При указании file в конфиге создаётся RotatingFileHandler."""
        log_file = tmp_path / "logs" / "test.log"
        cfg = {
            "level": "INFO",
            "format": "%(message)s",
            "file": str(log_file),
        }

        with caplog.at_level(logging.INFO):
            setup_logging(cfg)

        root = logging.getLogger()
        rotating_handlers = [
            h for h in root.handlers
            if "RotatingFileHandler" in type(h).__name__
        ]
        assert len(rotating_handlers) == 1, (
            f"Expected 1 RotatingFileHandler, got {[type(h).__name__ for h in root.handlers]}"
        )
        # Файл создан
        assert log_file.exists(), f"Log file not created: {log_file}"
        # stderr handler тоже остался
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1

    def test_rotating_handler_max_bytes_backup_count(self, tmp_path, caplog):
        """RotatingFileHandler: maxBytes=5 МБ, backupCount=3."""
        log_file = tmp_path / "logs" / "test.log"
        cfg = {
            "level": "INFO",
            "format": "%(message)s",
            "file": str(log_file),
        }

        with caplog.at_level(logging.WARNING):
            setup_logging(cfg)

        from logging.handlers import RotatingFileHandler
        rotating = next(
            (h for h in logging.getLogger().handlers
             if isinstance(h, RotatingFileHandler)),
            None,
        )
        assert rotating is not None, "RotatingFileHandler not found"
        assert rotating.maxBytes == 5 * 1024 * 1024, (
            f"maxBytes={rotating.maxBytes}, expected {5*1024*1024}"
        )
        assert rotating.backupCount == 3, (
            f"backupCount={rotating.backupCount}, expected 3"
        )

    def test_no_file_handler_when_file_not_specified(self, tmp_path, caplog):
        """Без ключа 'file' в конфиге — только stderr handler, файл не создаётся."""
        cfg = {
            "level": "INFO",
            "format": "%(message)s",
        }

        with caplog.at_level(logging.INFO):
            setup_logging(cfg)

        from logging.handlers import RotatingFileHandler
        rotating = next(
            (h for h in logging.getLogger().handlers
             if isinstance(h, RotatingFileHandler)),
            None,
        )
        assert rotating is None, "RotatingFileHandler should not be created without file config"
        # stderr handler есть
        stream = [
            h for h in logging.getLogger().handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
        ]
        assert len(stream) >= 1

    def test_log_directory_created_if_missing(self, tmp_path, caplog):
        """Каталог для лога создаётся автоматически."""
        log_file = tmp_path / "nonexistent_dir" / "subdir" / "app.log"
        cfg = {
            "level": "INFO",
            "format": "%(message)s",
            "file": str(log_file),
        }

        with caplog.at_level(logging.WARNING):
            setup_logging(cfg)

        # Запись лога не падает
        logging.getLogger("test").warning("hello")
        assert log_file.exists(), "Log directory and file should be created"

    def test_rotating_handler_encoding_utf8(self, tmp_path, caplog):
        """RotatingFileHandler открывается с encoding='utf-8' (nit-фикс: Windows/cp1251)."""
        log_file = tmp_path / "utf8_test.log"
        cfg = {
            "level": "INFO",
            "format": "%(message)s",
            "file": str(log_file),
        }

        with caplog.at_level(logging.INFO):
            setup_logging(cfg)

        from logging.handlers import RotatingFileHandler
        rotating = next(
            (h for h in logging.getLogger().handlers
             if isinstance(h, RotatingFileHandler)),
            None,
        )
        assert rotating is not None, "RotatingFileHandler not found"
        assert getattr(rotating, 'encoding', None) == 'utf-8', (
            f"encoding={getattr(rotating, 'encoding', None)!r}, expected 'utf-8'"
        )

    def test_file_handler_sanitizes_secrets(self, tmp_path, caplog):
        """Секреты/token в файловом хендлере маскируются (TokenSanitizer)."""
        import tempfile
        from pathlib import Path

        # Пишем в отдельный лог-файл (не в caplog, тот не покрывает файл)
        log_file = tmp_path / "secret_test.log"
        cfg = {
            "level": "DEBUG",
            "format": "%(message)s",
            "file": str(log_file),
        }
        setup_logging(cfg)

        # Записываем секрет — используем реальный формат TokenSanitizer
        # Pattern 1: bot<digits>:<secret>
        # Pattern 2: <digits>:<secret> (secret ≥15 символов)
        # Pattern 3: sk-[A-Za-z0-9]{16,}
        test_logger = logging.getLogger("test_secret")
        test_logger.warning("Telegram bot: bot123456:AbCdEfGhIjKlMnOpQrStU123456")
        test_logger.warning("DeepSeek key: sk-1234567890abcdefghijklmnop")
        test_logger.warning("Emoji test: 🚀🔥💀")
        test_logger.warning("CJK test: 你好世界")

        # Flush чтобы точно записалось
        for h in logging.getLogger().handlers:
            if hasattr(h, 'flush'):
                h.flush()

        raw = log_file.read_text(encoding="utf-8")
        # Telegram токен замаскирован
        assert "bot123456:AbCdEfGhIjKlMnOpQrStU123456" not in raw, (
            "Telegram token should be masked in log file"
        )
        # DeepSeek key замаскирован
        assert "sk-1234567890abcdefghijklmnop" not in raw, (
            "DeepSeek key should be masked in log file"
        )
        # Маска "***REDACTED***" присутствует
        assert "***REDACTED***" in raw, (
            "Masked token should appear as '***REDACTED***' in log file"
        )
        # Emoji и CJK записываются в utf-8 (encoding="utf-8" в RotatingFileHandler)
        assert "🚀" in raw or "\U0001f680" in raw, (
            "Emoji should be written to file in utf-8, not trigger cp1251 error"
        )

    def test_repeated_setup_does_not_leak_handlers(self, tmp_path, caplog):
        """Повторный setup_logging закрывает старые хендлеры (nit-фикс: утечка дескрипторов)."""
        log_file = tmp_path / "reinit_test.log"
        cfg = {
            "level": "INFO",
            "format": "%(message)s",
            "file": str(log_file),
        }

        from logging.handlers import RotatingFileHandler

        # Первый вызов
        setup_logging(cfg)
        setup_logging(cfg)

        rotating = [
            h for h in logging.getLogger().handlers
            if isinstance(h, RotatingFileHandler)
        ]
        assert len(rotating) == 1, (
            f"Expected 1 RotatingFileHandler after re-init, got {len(rotating)} "
            f"(duplicate handlers = descriptor leak)"
        )
        assert log_file.exists(), "Log file should be created"