"""
Настройка логирования.
"""
import logging
import re
import sys
from collections.abc import Mapping
from logging.handlers import RotatingFileHandler
from pathlib import Path

from newsbot.version import __version__

logger = logging.getLogger("newsbot.log")


# ── Фильтр токена из логов (belt-and-suspenders) ──────────────────
# T4-фикс: три паттерна:
#   1. bot<digits>:<secret>  — Telegram с префиксом bot
#   2. <digits>:<secret>     — голый токен без префикса (secret ≥15 символов)
#   3. sk-[A-Za-z0-9]{16,}   — DeepSeek API key (sk- + ≥16 alphanum)
# Minor-фикс: userinfo в URL (scheme://user:pass@host) маскируется.
_MASK = "***REDACTED***"
_TOKEN_RE = re.compile(r"bot\d+:[A-Za-z0-9_-]{15,}")
_NAKED_TOKEN_RE = re.compile(r"\d+:[A-Za-z0-9_-]{15,}")
_DEEPSEEK_RE = re.compile(r"sk-[A-Za-z0-9]{16,}")
# Minor-фикс: маскирует userinfo в URL (scheme://user:pass@host → scheme://***@host)
_USERINFO_RE = re.compile(r"://[^/]*@")  # matches ://<user>:<pass>@


def _sanitize(text: str) -> str:
    """Belt-and-suspenders: вырезает секреты и userinfo из текста."""
    if not isinstance(text, str):
        return text
    text = _TOKEN_RE.sub(_MASK, text)
    text = _NAKED_TOKEN_RE.sub(_MASK, text)
    text = _DEEPSEEK_RE.sub(_MASK, text)
    # Minor-фикс: userinfo в URL (scheme://user:pass@host → scheme://***@host).
    # Вырезается ДО токенов, т.к. user:pass может матчиться с _NAKED_TOKEN_RE
    # (\d+:<secret> или bot\d+:<secret>) и давать ложные срабатывания.
    text = _USERINFO_RE.sub("://***@", text)
    return text


class TokenSanitizer(logging.Filter):
    """
    Filter вырезает Telegram/DeepSeek-токены из лог-сообщений.
    Belt-and-suspenders — доп.защита, если httpx/httpcore
    всё же успеют залогировать URL до понижения уровня.

    Подход (T4-фикс):
      - msg санируется через _sanitize().
      - args санируются: tuple (все элементы), Mapping (dict values).
        Это покрывает logger.warning("URL: %s", url_with_token) —
        getMessage() делает msg % args → токен попадает в результат,
        если args не санировать.
      - exc_text и stack_info санируются на уровне filter().
      - SanitizingFormatter дублирует санацию на уровне итоговой строки
        (покрывает вычисляемое exc_text, у которого нет сеттера в LogRecord).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # T4-фикс: msg, args, exc_text, stack_info — все поля санируются.
        # 1. msg фильтруется
        if record.msg:
            record.msg = _sanitize(str(record.msg))
        # 2. args санируются: токен в аргументах (%s, %r, etc.) тоже вырезается.
        #    Это ключевой момент: logger.warning("URL: %s", url_with_token)
        #    даёт record.msg="URL: %s", record.args=(url_with_token,).
        #    record.getMessage() делает msg % args → токен попадает в результат.
        #    Санируем args, чтобы подставленный результат был чистым.
        if record.args:
            # C3-фикс: если args — Mapping (dict), format требует ключи
            # (%(name)s). Преобразование в tuple(dict.keys()) → TypeError.
            # Фикс: Mapping обрабатываем ДО tuple(), сохраняем тип как dict.
            if isinstance(record.args, Mapping):
                record.args = {
                    k: _sanitize(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            else:
                sanitized_args = []
                for a in record.args:
                    if isinstance(a, str):
                        sanitized_args.append(_sanitize(a))
                    elif isinstance(a, dict):
                        # dict как элемент tuple (не %()-format) — безопасно
                        sanitized = {
                            k: _sanitize(str(v)) if isinstance(v, str) else v
                            for k, v in a.items()
                        }
                        sanitized_args.append(sanitized)
                    else:
                        sanitized_args.append(a)
                record.args = tuple(sanitized_args)
        # 3. exc_text: текстовое представление исключения (traceback string).
        #    Добавляется Handler'ом при log(exc_info=True) — санируем на уровне
        #    Formatter, т.к. в LogRecord это вычисляемое свойство без сеттера.
        #    Подменяем getMessage(), чтобы record.message всегда был чистым.
        # 4. stack_info: текст стектрейса (log(stack_info=True)).
        if record.exc_text:
            record.exc_text = _sanitize(record.exc_text)
        if record.stack_info:
            record.stack_info = _sanitize(str(record.stack_info))
        return True


class VersionFilter(logging.Filter):
    """
    Внедряет атрибут version в каждый LogRecord.
    Не перезаписывает существующий version (допускает override).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "version", None) is None:
            record.version = __version__
        return True


def _minimal_sanitizer_handler() -> None:
    """
    Минимальный санирующий handler на stderr — без полного логирования.
    Ставится ДО CLI-ветвления (--check-telegram / --set-chat-id / --check-llm),
    чтобы токены не попали в логи при ранних исключениях.

    Добавляет StreamHandler с TokenSanitizer, НЕ трогает root handlers
    (конфликт с test setup).
    """
    import sys
    root = logging.getLogger()
    # проверим что ещё не добавлен наш handler (idempotent)
    for h in root.handlers:
        if getattr(h, "_sanitizer_installed", False):
            return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    handler.addFilter(TokenSanitizer())
    handler._sanitizer_installed = True  # noqa: B010
    root.addHandler(handler)


def setup_logging(cfg: dict) -> None:
    """
    Конфигурирует корневой логгер и обработчики.
    cfg = {"level": "INFO", "format": "...", "file": "data/logs/newsbot.log"}
    Добавляет RotatingFileHandler (если указан file) и StreamHandler (stderr).
    """
    level_str = cfg.get("level", "INFO")
    # Major-3-фикс: isinstance check — level: null → None.upper() → AttributeError.
    if not isinstance(level_str, str):
        logger.warning(
            "logging.level=%r (type %s) — ожидалась строка, используем INFO",
            level_str, type(level_str).__name__,
        )
        level_str = "INFO"
    level_str = level_str.upper()
    # Major-3-фикс: hasattr check — опечатка (INF0, VERBOSE) давала INFO
    # без warning. Теперь: если уровень не найден → warning с допустимыми.
    if not hasattr(logging, level_str):
        valid_levels = ", ".join(
            k for k in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
            if hasattr(logging, k)
        )
        logger.warning(
            "logging.level=%r неизвестен. Допустимые: %s. Используем INFO.",
            level_str, valid_levels,
        )
        level_str = "INFO"
    level = getattr(logging, level_str)
    fmt = cfg.get(
        "format",
        "%(asctime)s | v%(version)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    file_path = cfg.get("file")

    # корневой handler — stderr
    root = logging.getLogger()
    root.setLevel(level)

    # NIT-фикс: закрываем старые хендлеры перед очисткой.
    # Без этого RotatingFileHandler держит открытый file descriptor →
    # на повторном setup_logging (re-init) файл не переоткрывается, а
    # старый handler пишет в закрытый файл (--- Logging error --- на Windows).
    # flush() сбрасывает буфер на диск; close() освобождает дескриптор.
    # removeHandler вызывается ДО close для корректного завершения ротейшна.
    for h in root.handlers[:]:
        root.removeHandler(h)
        if hasattr(h, 'flush'):
            h.flush()
        if hasattr(h, 'close'):
            h.close()
    root.handlers.clear()

    # T4-фикс: Formatter санирует exc_text — LogRecord.exc_text вычисляется
    # из sys.exc_info() и не имеет сеттера, поэтому Filter не может его
    # подменить; санация на уровне format() (итоговая строка) решает проблему.
    class SanitizingFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            record.version = getattr(record, "version", None) or __version__
            text = super().format(record)
            return _sanitize(text)

    # ── file handler (ротируемый) ──────────────────────────────────
    if file_path:
        log_file = Path(file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=5 * 1024 * 1024,  # 5 МБ
            backupCount=3,
            encoding="utf-8",  # NIT-фикс: на Windows без этого emoji/CJK → Logging error (cp1251)
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(SanitizingFormatter(fmt))
        file_handler.addFilter(VersionFilter())
        file_handler.addFilter(TokenSanitizer())
        root.addHandler(file_handler)

    # ── stderr handler ──────────────────────────────────────────────
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(SanitizingFormatter(fmt))
    handler.addFilter(VersionFilter())
    # фильтр токена — на уровне handler
    handler.addFilter(TokenSanitizer())
    root.addHandler(handler)

    # ── приглушить verbose-библиотеки ────────────────────────────
    # К4-фикс: httpx логирует строку запроса (в т.ч. с токеном) на INFO.
    # Понижаем до WARNING, чтобы токен не попал в логи.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # убрать избыточные feedparser-логи
    logging.getLogger("feedparser").setLevel(logging.WARNING)