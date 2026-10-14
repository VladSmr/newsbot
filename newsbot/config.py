"""
Загрузка конфигурации: config.yaml + .env.
.env покрывает telegram.token, telegram.chat_id, llm.api_key.

Поиск корня проекта:
  — восхождение от __file__ вверх, ищем setup.py / config.yaml / .env
    (маркеры никогда не попадают в site-packages).
  — если не найдены: тихий fallback на предок каталога пакета
    (для newsbot/newsbot/ → проектный корень).
"""
import copy
import logging
import os as _os
from pathlib import Path
from typing import Any, Dict

import yaml

from .env import telegram_token, deepseek_api_key, get_env, _set_env_project_root

logger = logging.getLogger("newsbot.config")


def _find_project_root() -> Path:
    """
    Ищет корень проекта восхождением от __file__ вверх.

    Маркеры проектного корня (НЕ копируются в site-packages):
      — setup.py          (всегда в корне проекта)
      — config.yaml       (пользовательский файл в корне)
      — .env              (пользовательский файл в корне)
    Любой из маркеров — достаточное условие для остановки.
    Предок каталога пакета (newsbot/newsbot/) используется
    как последний тихий fallback.
    """
    _pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
    _parent_dir = _os.path.dirname(_pkg_dir)  # запасной fallback
    # Надёжные маркеры проектного корня (НЕ копируются в site-packages):
    #   setup.py   — всегда в корне, никогда внутри пакета
    #   config.yaml / .env — пользовательские файлы в корне проекта
    # ДОСТАТОЧНО одного маркера для остановки восхождения.
    for _ in range(10):
        _p = Path(_pkg_dir)
        if (
            (_p / "setup.py").exists()
            or (_p / "config.yaml").exists()
            or (_p / ".env").exists()
        ):
            return _p
        _parent = _os.path.dirname(_pkg_dir)
        if _parent == _pkg_dir:
            break
        _pkg_dir = _parent
    # Не нашли маркеры — значит пакет установлен и корень недоступен.
    # Используем предок каталога пакета:
    #   newsbot/newsbot/ → предок = newsbot/ = проектный корень ✅
    #   site-packages/newsbot/ → предок = site-packages/ (edge case,
    #   пользователь должен запускать из проектного каталога).
    return Path(_parent_dir)


# Синхронизируем root с env.py (lazy, idempotent)
_PROJECT_ROOT = _find_project_root()
_set_env_project_root(_PROJECT_ROOT)

DEFAULT_CONFIG: Dict[str, Any] = {
    "poll_interval_minutes": 15,
    "database": {"path": "data/newsbot.db", "retention_days": 90, "retry_threshold_sec": 60},
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    },
    "telegram": {"token": "", "chat_id": "", "poll_timeout": 25},
    "digest": {
        "enabled": True,
        "times": ["09:00", "21:00"],
        "max_items": 100,
        "max_items_per_message": 20,
        "on_empty": "skip",
    },
    "llm": {
        "enabled": True,
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        "api_key": "",
        "temperature": 0,
        "timeout_sec": 60,
        "max_retries": 3,
        "push_threshold": 7,
        "digest_threshold": 5,
        "system_prompt": "",
        "balance": {
            "enabled": True,
            "threshold": 2.0,
            "check_interval_hours": 24,
        },
        "max_per_cycle": 150,
    },
    # Фаза 2: обратная связь (кнопки 👍/👎)
    "feedback": {
        "enabled": True,
        "max_examples": 10,     # примеров в few-shot промпте
        "retention_days": 180,   # хранение feedback_targets (feedback хранится дольше)
    },
}


def load_config(config_path: str = "") -> Dict[str, Any]:
    """
    Загружает config.yaml.
    По умолчанию: config.yaml в проектном корне (_PROJECT_ROOT).
    Если не найден — тихий fallback на cwd()/config.yaml
    (пользователь запускает бота из проектного каталога).
    """
    if not config_path:
        config_path = str(_PROJECT_ROOT / "config.yaml")
        # Тихий fallback: если конфиг не найден в _PROJECT_ROOT,
        # пробуем cwd() (пользователь мог запустить из проектного каталога)
        if not Path(config_path).exists():
            config_path = str(Path.cwd() / "config.yaml")

    p = Path(config_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        logger.info("Конфиг загружен из %s (%d фидов)", p, len(raw.get("sources", [])))
    else:
        logger.warning("config.yaml не найден по пути %s, используем defaults", p)
        raw = {}

    # deep-merge: copy.deepcopy защищает от алиасинга вложенных dict
    cfg = _deep_merge(copy.deepcopy(DEFAULT_CONFIG), raw)

    # ── Telegram: .env поверх config.yaml ──
    # .env читается только для telegram.* — остальное из config.yaml
    env_token = telegram_token()
    env_chat_id = get_env("TELEGRAM_CHAT_ID", "")

    tg_section = cfg.setdefault("telegram", {})
    if env_token:
        tg_section["token"] = env_token
    if env_chat_id:
        tg_section["chat_id"] = env_chat_id

    # ── LLM: .env поверх config.yaml ──
    env_llm_key = deepseek_api_key()
    if env_llm_key:
        llm_section = cfg.setdefault("llm", {})
        llm_section["api_key"] = env_llm_key

    return cfg


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Рекурсивный merge overlay поверх base (base предварительно deepcopy)."""
    result = base
    for k, v in overlay.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = _deep_merge(result[k], v)  # type: ignore
        else:
            result[k] = v
    return result