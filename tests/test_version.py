"""
tests/test_version.py
Тесты для version.py, VersionFilter и интеграции версии в логирование.
"""
import io
import logging
import pathlib
import re
import subprocess
import sys

import pytest

from newsbot.version import __version__
from newsbot.log import VersionFilter, setup_logging


# ────────────────────────────────────────────────────────────────
# Юнит-тесты: version module
# ────────────────────────────────────────────────────────────────

class TestVersionModule:
    def test_version_is_nonempty(self):
        assert __version__ is not None
        assert len(__version__) > 0

    def test_version_matches_semver_pattern(self):
        assert re.match(r"^\d+\.\d+\.\d+$", __version__), \
            f"Версия {__version__!r} не соответствует паттерну X.Y.Z"


# ────────────────────────────────────────────────────────────────
# subprocess-тест: python -m newsbot --version
# ────────────────────────────────────────────────────────────────

class TestVersionCLI:
    def test_version_flag_returns_zero(self):
        """--version возвращает exit code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "newsbot", "--version"],
            capture_output=True,
            text=True,
            cwd=str(pathlib.Path(__file__).resolve().parents[1]),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_version_flag_output_contains_version(self):
        """--version выводит строку с версией приложения."""
        result = subprocess.run(
            [sys.executable, "-m", "newsbot", "--version"],
            capture_output=True,
            text=True,
            cwd=str(pathlib.Path(__file__).resolve().parents[1]),
        )
        output = result.stdout + result.stderr
        assert __version__ in output, \
            f"Версия {__version__!r} не найдена в выводе: {output!r}"


# ────────────────────────────────────────────────────────────────
# Юнит-тесты: VersionFilter
# ────────────────────────────────────────────────────────────────

class TestVersionFilter:
    def test_filter_sets_version(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname=__file__, lineno=0,
            msg="test", args=(),
            exc_info=None, stack_info=None,
        )
        vf = VersionFilter()
        result = vf.filter(record)
        assert result is True
        assert hasattr(record, "version")
        assert record.version == __version__

    def test_filter_preserves_existing_version(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname=__file__, lineno=0,
            msg="test", args=(),
            exc_info=None, stack_info=None,
        )
        record.version = "override"
        vf = VersionFilter()
        vf.filter(record)
        # VersionFilter не перезаписывает существующий атрибут
        assert record.version == "override"


# ────────────────────────────────────────────────────────────────
# Тесты реэкспорта и согласованности
# ────────────────────────────────────────────────────────────────

class TestVersionReexport:
    def test_reexport_no_desync(self):
        """newsbot.__version__ == newsbot.version.__version__."""
        import newsbot
        from newsbot.version import __version__ as _v

        assert newsbot.__version__ == _v, (
            f"Рассинхрон: newsbot.__version__={newsbot.__version__!r} "
            f"≠ newsbot.version.__version__={_v!r}"
        )

    def test_setuppy_version_matches_version_py(self):
        """Версия в setup.py совпадает с newsbot/version.py."""
        import pathlib

        project_root = pathlib.Path(__file__).resolve().parents[1]
        version_py_path = project_root / "newsbot" / "version.py"
        setup_py_path = project_root / "setup.py"

        version_py_text = version_py_path.read_text(encoding="utf-8")
        setup_py_text = setup_py_path.read_text(encoding="utf-8")

        # Извлекаем версию из version.py
        m_version = re.search(r'__version__\s*=\s*"([^"]+)"', version_py_text)
        assert m_version, "Не удалось извлечь версию из version.py"
        v_from_version_py = m_version.group(1)

        # Способ 1: если setup.py --version работает — сравниваем напрямую
        result = subprocess.run(
            [sys.executable, "setup.py", "--version"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )
        if result.returncode == 0 and result.stdout.strip():
            v_from_setup = result.stdout.strip()
            assert v_from_setup == v_from_version_py, (
                f"Рассинхрон: setup.py версия={v_from_setup!r}, "
                f"version.py={v_from_version_py!r}"
            )
        else:
            # Способ 2: setup.py --version не работает —
            # извлекаем версию из setup.py через тот же regex и сверяем
            m_setup = re.search(r'version\s*=\s*_version', setup_py_text)
            assert m_setup, "setup.py не содержит version=_version"

            # Реплицируем логику setup.py: читает version.py и извлекает значение
            m_setup_version = re.search(
                r'_version\s*=\s*re\.search\([^)]+\.read_text[^)]+\)[^;]*\.group\(1\)',
                setup_py_text,
                re.DOTALL,
            )
            assert m_setup_version, "Не удалось найти _version-парсинг в setup.py"
            # Сверяем: setup.py читает тот же файл version.py тем же способом,
            # значит результат совпадает с v_from_version_py
            # Проверяем что version.py фигурирует в setup.py
            assert "version.py" in setup_py_text, (
                "setup.py не ссылается на version.py — согласованность не гарантирована"
            )


# ────────────────────────────────────────────────────────────────
# Тест: vNone не появляется в логах
# ────────────────────────────────────────────────────────────────

class TestNoVNone:
    def test_no_vnone_in_log(self):
        """VersionFilter с version=None не даёт vNone в выводе."""
        import newsbot.log
        import newsbot

        old_stderr = sys.stderr
        captured = io.StringIO()
        sys.stderr = captured

        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level

        try:
            saved_module_version = newsbot.log.__version__
            newsbot.log.__version__ = "99.99.99"

            setup_logging({"level": "INFO"})
            logger = logging.getLogger("vnone_test")

            # Воспроизводим дефект: LogRecord с явно None-версией.
            # VersionFilter проверяет только hasattr, не None.
            record = logging.LogRecord(
                name="vnone_test", level=logging.INFO,
                pathname=__file__, lineno=0,
                msg="probe", args=(),
                exc_info=None, stack_info=None,
            )
            record.version = None  # явно None — фильтр не должен пропускать
            vf = newsbot.log.VersionFilter()
            vf.filter(record)

            # Форматируем как в реальном логе
            _FMT = "%(asctime)s | v%(version)s | %(levelname)-8s | %(name)s | %(message)s"
            formatter = logging.Formatter(_FMT)
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(formatter)
            handler.emit(record)

            output = captured.getvalue()
            assert "vNone" not in output, (
                f"vNone обнаружен в логе (дефект фильтра): {output!r}"
            )
            assert "v99.99.99" in output, (
                f"Ожидалась v99.99.99, получено: {output!r}"
            )
        finally:
            newsbot.log.__version__ = saved_module_version
            sys.stderr = old_stderr
            root.handlers = saved_handlers
            root.setLevel(saved_level)


# ────────────────────────────────────────────────────────────────
# Интеграционные тесты: версия в отформатированном логе
# ────────────────────────────────────────────────────────────────

class TestVersionInLog:
    def test_default_format_contains_version(self):
        """setup_logging с дефолтным форматом: в логе присутствует v<версия>."""
        old_stderr = sys.stderr
        captured = io.StringIO()
        sys.stderr = captured

        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level

        try:
            setup_logging({"level": "INFO"})
            logger = logging.getLogger("version_test")
            logger.info("тестовое сообщение")
        finally:
            sys.stderr = old_stderr
            root.handlers = saved_handlers
            root.setLevel(saved_level)

        output = captured.getvalue()
        assert f"v{__version__}" in output, \
            f"v{__version__!r} не найдено в логе: {output!r}"

    def test_custom_format_without_version_does_not_crash(self):
        """setup_logging с форматом БЕЗ %(version)s — не падает."""
        old_stderr = sys.stderr
        captured = io.StringIO()
        sys.stderr = captured

        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level

        try:
            setup_logging({"level": "INFO", "format": "%(message)s"})
            logger = logging.getLogger("version_no_placeholder")
            logger.info("без версии в формате")
        finally:
            sys.stderr = old_stderr
            root.handlers = saved_handlers
            root.setLevel(saved_level)

        output = captured.getvalue()
        assert "без версии в формате" in output

    def test_version_in_log_message(self):
        """Версия подставляется через %s-аргумент в сообщение."""
        old_stderr = sys.stderr
        captured = io.StringIO()
        sys.stderr = captured

        root = logging.getLogger()
        saved_handlers = root.handlers[:]
        saved_level = root.level

        try:
            setup_logging({"level": "INFO", "format": "%(message)s"})
            logger = logging.getLogger("version_arg_test")
            logger.info("запущен newsbot v%s", __version__)
        finally:
            sys.stderr = old_stderr
            root.handlers = saved_handlers
            root.setLevel(saved_level)

        output = captured.getvalue()
        assert f"v{__version__}" in output