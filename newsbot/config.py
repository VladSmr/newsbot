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
    "database": {"path": "data/newsbot.db", "retention_days": 1, "retry_threshold_sec": 60,
                   "deferred_retention_days": 3, "recovery_new_limit": 500,
                   "max_retries": 8},
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s | v%(version)s | %(levelname)-8s | %(name)s | %(message)s",
    },
    "telegram": {"token": "", "chat_id": "", "poll_timeout": 25},
    "digest": {
        "enabled": True,
        "cluster_duplicates": True,  # финальная LLM-кластеризация deferred перед отправкой
        "times": ["08:00", "20:00"],
        "max_items": 30,
        "max_items_per_message": 20,
        "on_empty": "skip",
        "max_age_hours": 24,
        "bootstrap_window_hours": 24,
        "max_per_location": 2,  # макс. новостей на локацию/день в дайджесте
        "category_min": {
            "cinema": 2,
            "games": 1,
            "space": 1,
            "tech": 1,
            "cyber": 1,
            "science": 1,
        },
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
        "off_peak_only": False,
        "batch_size": 10,
        "max_tokens": 4096,
        "llm_dedup": True,
        "push_urgent_only": True,
        # Pricing: USD per 1M tokens (verify current DeepSeek prices)
        "pricing": {
            "deepseek-flash": {"input": 0.27, "output": 1.10},   # placeholder
            "deepseek-v4-pro": {"input": 0.50, "output": 2.00}, # placeholder
        },
        "daily_cost_limit": 0.50,  # USD per day; 0 = disabled
    },
    # Фаза 2: обратная связь (кнопки 👍/👎)
    "feedback": {
        "enabled": True,
        "digest_enabled": False, # вкл/выкл фидбэк в дайджесте (дефолт false — нет кнопок 👍/👎 в дайджесте)
        "max_examples": 10,     # примеров в few-shot промпте
        "retention_days": 180,   # хранение feedback_targets (feedback хранится дольше)
    },
    # Кросс-источниковая дедупликация
    "dedup": {
        "cross_source": True,   # уровень 1: глобальный точный дедуп по title_hash (без source)
        "fuzzy": True,          # уровень 2: нечёткое сравнение (SequenceMatcher), только при cross_source=true
        "threshold": 0.95,      # порог SequenceMatcher ratio (0..1, clamp автоматический)
        "token_threshold": 0.8,  # порог Jaccard значащих токенов (0..1, clamp автоматический)
        "window_days": 3,       # окно сравнения для fuzzy и cross_source
        "llm_recent_limit": 20,  # недавних заголовков для LLM-дедупа
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
        # Minor 5-фикс: YAML-корень должен быть словарём. Скаляр/список
        # (например случайная строка в конфиге) не должен ронять загрузку
        # TypeError'ом в _deep_merge/len().
        if not isinstance(raw, dict):
            logger.warning(
                "Корень config.yaml должен быть словарём, получено %s — "
                "используем defaults",
                type(raw).__name__,
            )
            raw = {}
        # Minor 5-фикс: sources: null → len(None) падал TypeError.
        logger.info(
            "Конфиг загружен из %s (%d фидов)",
            p, len(raw.get("sources") or []),
        )
    else:
        logger.warning("config.yaml не найден по пути %s, используем defaults", p)
        raw = {}

    # deep-merge: copy.deepcopy защищает от алиасинга вложенных dict
    cfg = _deep_merge(copy.deepcopy(DEFAULT_CONFIG), raw)

    # Minor 5-фикс + Major 2-фикс (а): валидация числовых полей llm-секции
    # в единой точке загрузки конфига — невалидное значение → warning
    # и fallback на дефолт из DEFAULT_CONFIG.
    _validate_llm_numeric_fields(cfg)

    # m1 + Minor (guard не-dict секций): YAML-секция, заданная как `llm:`
    # (null) ИЛИ скаляром (`llm: deepseek`, `telegram: foo`), не должна
    # затирать дефолтный dict — иначе последующие cfg["llm"].get(...)
    # падают с AttributeError. Для известных секций-словарей из
    # DEFAULT_CONFIG восстанавливаем дефолт (с warning, не молча).
    for section, default_value in DEFAULT_CONFIG.items():
        if not isinstance(default_value, dict):
            continue
        if not isinstance(cfg.get(section), dict):
            logger.warning(
                "config.yaml: секция %r должна быть словарём, получено %s — "
                "используем defaults",
                section, type(cfg.get(section)).__name__,
            )
            cfg[section] = copy.deepcopy(default_value)

    # Аналогично для вложенной секции llm.balance: pipeline читает её как
    # config["llm"].get("balance").get(...) — скаляр сломал бы это.
    llm_section = cfg.get("llm")
    if isinstance(llm_section, dict) and not isinstance(
        llm_section.get("balance"), dict
    ):
        logger.warning(
            "config.yaml: секция 'llm.balance' должна быть словарём, "
            "получено %s — используем defaults",
            type(llm_section.get("balance")).__name__,
        )
        llm_section["balance"] = copy.deepcopy(DEFAULT_CONFIG["llm"]["balance"])

    # Прочие None-значения верхнего уровня → {} (m1), чтобы .get-цепочки
    # по неизвестным секциям не падали на None.
    for key in list(cfg.keys()):
        if cfg[key] is None:
            cfg[key] = {}

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


def _validate_llm_numeric_fields(cfg: Dict[str, Any]) -> None:
    """
    Валидация числовых полей секции ``llm`` при загрузке конфига.

    Единая точка (Major 2-фикс (а)): невалидное значение (например
    ``push_threshold: "high"``) → logger.warning и fallback на дефолт
    из DEFAULT_CONFIG. Фильтр после этого создаётся и работает, а не
    отключается целиком из-за опечатки.

    Правит cfg in-place.
    """
    llm_cfg = cfg.get("llm")
    if not isinstance(llm_cfg, dict):
        return

    defaults = DEFAULT_CONFIG["llm"]
    # (путь в llm-секции, тип; int — не bool)
    spec = (
        (("push_threshold",), int),
        (("digest_threshold",), int),
        (("max_retries",), int),
        (("max_tokens",), int),
        (("timeout_sec",), float),
        (("temperature",), float),
        (("daily_cost_limit",), float),
        (("batch_size",), int),
        (("max_per_cycle",), int),
        (("balance", "threshold"), float),
        (("balance", "check_interval_hours"), int),
    )

    for path, expected in spec:
        container: Any = llm_cfg
        ok = True
        for part in path[:-1]:
            container = container.get(part) if isinstance(container, dict) else None
            if not isinstance(container, dict):
                ok = False
                break
        if not ok:
            continue

        key = path[-1]
        if key not in container:
            continue

        raw = container[key]
        default = defaults
        for part in path:
            default = default.get(part) if isinstance(default, dict) else None

        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            converted = None
        else:
            try:
                if expected is int:
                    converted = int(raw) if float(int(raw)) == float(raw) else None
                else:
                    value = float(raw)
                    converted = value if value == value else None  # отсекает NaN
            except (TypeError, ValueError, OverflowError):
                converted = None

        if converted is None:
            logger.warning(
                "llm.%s=%r — ожидалось %s, используется дефолт %r",
                ".".join(path), raw,
                "целое число" if expected is int else "число",
                default,
            )
            container[key] = default
        else:
            # Нормализуем к числу (например "9" → 9), чтобы потребители
            # конфига получали уже валидный тип.
            container[key] = converted


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Рекурсивный merge overlay поверх base (base предварительно deepcopy)."""
    result = base
    # Minor 5-фикс: YAML-корень не-dict (скаляр/список) — не падаем.
    if not isinstance(overlay, dict):
        logger.warning(
            "_deep_merge: overlay не словарь (%s) — игнорируем",
            type(overlay).__name__,
        )
        return result
    for k, v in overlay.items():
        if v is None and isinstance(result.get(k), dict):
            # m1: `section:` без значения (null) не затирает дефолтную секцию —
            # иначе .get-цепочки по секции падают с AttributeError.
            continue
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = _deep_merge(result[k], v)  # type: ignore
        else:
            result[k] = v
    return result