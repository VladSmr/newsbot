"""
Мини-парсер .env-файла.
Без зависимостей (не использует python-dotenv), парсит KEY=VALUE.
Поддерживает:
  - BOM (снимается)
  - кавычки вокруг значения: KEY="value" или KEY='value'
  - inline-комментарии: KEY=value # comment (B9: только ВНЕ кавычек)
  - export-префикс: export KEY=value
  - пустые строки, комментарии (#), строки без '=' — игнорируются
  - Windows-пути с бэкслэшами (B9: posix=False, не съедаем \\)

КОРНЕВОЙ ПОИСК: чтобы найти .env при установке (site-packages),
config.py вызывает _set_env_project_root() после определения корня.
Это должно произойти ДО первого вызова get_env/telegram_token/deepseek_api_key.
"""
import logging
import os as _os
import re
import shlex
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("newsbot.env")

# ── Поиск корня проекта (fallback) ────────────────────────────────
# Начальное значение: предок каталога пакета (src-layout: newsbot/src/ → newsbot/).
# Переопределяется через _set_env_project_root() из config.py.
_FALLBACK_ROOT = Path(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_PROJECT_ROOT: Path = _FALLBACK_ROOT
_ENV_PATH: Optional[Path] = None  # станет Path после _set_env_project_root


def _set_env_project_root(root: Path) -> None:
    """
    Устанавливает путь к корню проекта (вызывается из config.py).
    После этого _ENV_PATH указывает на правильный .env.
    """
    global _PROJECT_ROOT, _ENV_PATH
    _PROJECT_ROOT = root
    _ENV_PATH = root / ".env"
    logger.debug("Проект root синхронизирован: %s", _PROJECT_ROOT)


def _env_path() -> Path:
    """Ленивое определение пути к .env (нужен root от config.py)."""
    global _ENV_PATH
    if _ENV_PATH is None:
        _ENV_PATH = _PROJECT_ROOT / ".env"
    return _ENV_PATH


# ── Парсер .env ───────────────────────────────────────────────────
_RE_LINE = re.compile(
    r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$"
)

_cache: Optional[Dict[str, str]] = None


def _strip_inline_comment(s: str) -> str:
    """
    Вырезает inline-комментарий (# ...) из строки,
    но НЕ трогает # внутри кавычек.

    B9-фикс (rev2): # считается комментарием ТОЛЬКО если:
      - позиция 0 (начало значения), ИЛИ
      - предшествующий символ — пробел/таб
    Без этого KEY=a#b без кавычек усекался до "a".

    Примеры:
      value  # comment     → value
      "a#b"           → "a#b"
      'x # y'         → 'x # y'
      "a#b" # comment  → "a#b"
      a#b              → a#b (без пробела перед # — часть значения)
      KEY=a#b          → a#b (unquoted, нет пробела — часть значения)
      KEY= value#tag   → value#tag (# без пробела перед — часть значения)
    """
    in_single = False
    in_double = False
    for i, ch in enumerate(s):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            # первый # вне кавычек — всё что после него комментарий
            # B9-фикс (rev2): # комментарием считается только если:
            #   - позиция 0 (начало значения), ИЛИ
            #   - предшествующий символ — пробел/таб
            # Без этого KEY=a#b без кавычек усекался до "a".
            if i == 0 or s[i - 1] in (" ", "\t"):
                return s[:i].rstrip()
    return s


def _parse_env(path: Path) -> Dict[str, str]:
    if not path.exists():
        logger.debug(".env не найден: %s", path)
        return {}
    raw = path.read_text(encoding="utf-8-sig")
    result: Dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _RE_LINE.match(stripped)
        if not m:
            continue
        key = m.group(1)
        rest = m.group(2)

        # B9-фикс: вырезаем inline-комментарий ВНЕ кавычек
        rest = _strip_inline_comment(rest)

        # B9-фикс: posix=False чтобы не съедались бэкслэши (Windows-пути).
        # posix=False сохраняет кавычки вокруг значения → снимаем их вручную.
        # m2-фикс: shlex.split() для unquoted-значений с пробелами ("a b c")
        # возвращает ['a','b','c'] — берём parts[0]='a' (обрезает). Лечим:
        # используем shlex.split() только для кавыченных значений.
        if rest.lstrip().startswith(('"', "'")):
            try:
                parts = shlex.split(rest, posix=False)
                raw_value = parts[0] if parts else rest.strip()
            except ValueError:
                raw_value = rest.strip()
        else:
            # Unquoted: берём rest.strip() целиком (пробелы сохраняются)
            raw_value = rest.strip()
        # M2-фикс: снимаем ровно ОДНУ парную кавычку с начала и конца,
        # а не ВСЕ кавычки по краям (было strip('"').strip("'") — усекало VALUE" → VALUE).
        value = raw_value
        if len(value) >= 2:
            if (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"):
                value = value[1:-1]
        result[key] = value
    return result


def load_env() -> Dict[str, str]:
    global _cache
    if _cache is None:
        _cache = _parse_env(_env_path())
    return _cache


def get_env(key: str, default: str = "") -> str:
    """
    Значение переменной.

    Приоритет (от высшего к низшему):
      1. os.environ  — удобно для Docker / env_file в compose
      2. .env-файл   — локальная разработка (Windows/macOS/Linux)
      3. default     — fallback

    Docker: ``env_file: .env`` в compose устанавливает переменные в окружение
    контейнера, откуда они подхватываются через os.environ.
    """
    # 1) системное окружение (приоритетнее — для Docker env_file)
    val = _os.environ.get(key)
    if val:
        return val
    # 2) файл .env (локальный запуск)
    return load_env().get(key, default)


def telegram_token() -> str:
    return get_env("TELEGRAM_BOT_TOKEN", "")


def telegram_chat_id() -> str:
    return get_env("TELEGRAM_CHAT_ID", "")


def deepseek_api_key() -> str:
    return get_env("DEEPSEEK_API_KEY", "")


def _write_env(key: str, value: str) -> None:
    """
    Перезаписывает KEY=VALUE в .env (пишет в _env_path(), не site-packages).

    N7-фикс: запрещены \n и " в value.
      - \n ломает формат .env.
      - " требует экранирования (\") и корректного разэкранирования при чтении.
        Парсер снимает ровно одну парную кавычку с краёв; экранированные
        \" не разэкранируются, round-trip ломается. Простейшее решение — запрет ".
      - chmod 600: на Unix-совместимых системах (ignore на Windows).
    """
    if "\n" in value:
        raise ValueError(
            f"Нельзя записать KEY с переводом строки в .env: "
            f"{key!r} содержит \\n"
        )
    if '"' in value:
        raise ValueError(
            f'Нельзя записать KEY с кавычкой (") в .env: {key!r} содержит "'
        )
    # квотируем, если нужно
    needs_quotes = (" " in value or "#" in value)
    if needs_quotes:
        # " внутри value уже запрещена выше
        written = f'{key}="{value}"'
    else:
        written = f"{key}={value}"

    path = _env_path()
    lines: List[str] = []
    found = False
    if path.exists():
        raw = path.read_text(encoding="utf-8-sig")
        for line in raw.splitlines():
            m = _RE_LINE.match(line.strip())
            if m and m.group(1) == key:
                lines.append(written)
                found = True
            else:
                lines.append(line.rstrip())
    if not found:
        lines.append(written)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # T2-11: chmod 600 — только на Unix (Windows игнорирует)
    try:
        import stat
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    except (OSError, AttributeError):
        pass  # Windows или restricted env

    global _cache
    _cache = None
    logger.info(".env обновлён: %s=***", key)