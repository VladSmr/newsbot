"""
Настройка логирования.
"""
import logging
import re
import sys
from collections.abc import Mapping


# ── Фильтр токена из логов (belt-and-suspenders) ──────────────────
# Вырезает bot<digits>:<alphanum> из любых log-сообщений.
# Порог {15,} чтобы ловить типичные токены ≥15 символов после botXXXXX:.
_TOKEN_RE = re.compile(r"bot\d+:[A-Za-z0-9_-]{15,}")


class TokenSanitizer(logging.Filter):
    """
    Filter вырезает Telegram-токены из лог-сообщений.
    Паттерн: bot<digits>:<alphanum_string_len>=15.
    Belt-and-suspenders — доп.защита, если httpx/httpcore
    всё же успеют залогировать URL до понижения уровня.

    Подход: msg и args санируются, getMessage() вызывает
    _LogRecord__original_getMessage или форматируется через санированный msg % args.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # 1. msg фильтруется
        if record.msg:
            record.msg = _TOKEN_RE.sub("bot***:", str(record.msg))
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
                    k: _TOKEN_RE.sub("bot***:", str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            else:
                sanitized_args = []
                for a in record.args:
                    if isinstance(a, str):
                        sanitized_args.append(_TOKEN_RE.sub("bot***:", a))
                    elif isinstance(a, dict):
                        # dict как элемент tuple (не %()-format) — безопасно
                        sanitized = {
                            k: _TOKEN_RE.sub("bot***:", str(v)) if isinstance(v, str) else v
                            for k, v in a.items()
                        }
                        sanitized_args.append(sanitized)
                    else:
                        sanitized_args.append(a)
                record.args = tuple(sanitized_args)
        return True


def setup_logging(cfg: dict) -> None:
    """
    Конфигурирует корневой логгер и обработчики.
    cfg = {"level": "INFO", "format": "..."}
    """
    level_str = cfg.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    fmt = cfg.get(
        "format",
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # корневой handler — stderr
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
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