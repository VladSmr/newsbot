"""
Конвейер: Collector → Dedup → save (state='new') →
  LLM-фильтр → set_state('pending') → delivery →
  set_state('delivered') | отлов ошибки (остаётся 'pending').

State-машина (state, state_at):
  new       — обнаружен, в candidates, ждёт обработки
  pending   — обработка начата (set_state перед send_batch)
  delivered — успешно обработан и доставлен
  skipped   — LLM отфильтровал (score < digest_threshold, retry не трогает)
  deferred  — score 5–6, отложен на дайджест (retry не трогает)
  failed    — терминальный; retry_count исчерпан или recovery на старте цикла
  digested  — терминальный; отправлен в дайджесте

Retry: get_stuck_pending() берёт только 'pending' старше
config.database.retry_threshold_sec (по умолч. 60 с).
Если item уже имеет llm_score — retry НЕ переоценивает.

Двойной send в одном цикле исключён:
  new → pending → delivery. При падении — pending остаётся;
  retry сработает только в СЛЕДУЮЩЕМ цикле.

M2: run_digest может задерживать основной цикл при большом числе
deferred-items (до max_items сообщений с ретраями Telegram API).
Это ожидаемое поведение; на время дайджеста основной цикл блокируется.
"""


import logging
import math
from datetime import datetime, timedelta, timezone as tz
from email.utils import parsedate_to_datetime as _parsedate_to_dt
from typing import Dict, List, Optional, Set

from .collector import FeedCollector, FeedResult, NewsItem
from .dedup import filter_new
from .delivery import ConsoleDelivery, TelegramDelivery, Delivery
from .delivery.telegram import ChatIdNotBoundError, BotBlockedError
from .digest import run_digest as _run_digest, build_digest_messages
from .version import __version__
from .llm import LLMFilter
from .llm.filter import FilterResult, route
from .llm.client import BalanceInfo, LLMError, UsageStats
from .offpeak import is_llm_allowed_now
from .storage import Storage

logger = logging.getLogger("newsbot.pipeline")


def _parse_bool(value) -> bool:
    """
    Явный парсер bool-значений из конфига.
    - bool: как есть
    - int: 1 → True, 0 → False (bool подкласс int → check ПЕРЕД int)
    - str: "true","1","yes","on" (case-insensitive) → True; всё остальное → False
    - прочее: False
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "on")
    return False


def _coerce_int(value, default: int = 0, context: str = "значение") -> int:
    """
    Приводит значение из конфига к int с fallback на default.

    YAML может отдать число строкой ("2"): `str <= int` и срез `seq[:str]`
    бросают TypeError ВНЕ try вокруг _run_digest → дайджест падал бы каждый
    цикл (тот же класс бага, что чинили в dedup). Некорректное значение —
    warning + default (без исключения).

    Разбор через float (консистентно с dedup._cfg_int):
    - "2.0"/"2.9" принимаются (раньше строка с точкой отвергалась → 0);
    - NaN/inf (валидный YAML `.nan`/`.inf`) → default, а не ValueError/
      OverflowError из int() ВНЕ try вокруг _run_digest;
    - нецелый float не усекается молча: warning + floor.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning(
            "Некорректное значение %s=%r — использую %d", context, value, default,
        )
        return default
    if not math.isfinite(parsed):
        logger.warning(
            "Некорректное значение %s=%r — не конечное число, использую %d",
            context, value, default,
        )
        return default
    floored = int(math.floor(parsed))
    if floored != parsed:
        logger.warning(
            "Некорректное значение %s=%r — нецелое число, округлено вниз до %d",
            context, value, floored,
        )
    return floored


def _normalize_category_min(raw) -> Dict[str, int]:
    """
    Нормализует category_min из YAML в Dict[str, int].

    Невалидное/нечисловое значение категории → warning + 0 (категория
    игнорируется): опечатка в конфиге не должна ни ронять дайджест
    (TypeError в сравнении/срезе вне try), ни выдумывать гарантию.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        logger.warning(
            "category_min: ожидался dict, получено %s — игнорирую",
            type(raw).__name__,
        )
        return {}
    normalized: Dict[str, int] = {}
    for cat, value in raw.items():
        normalized[str(cat)] = _coerce_int(
            value, default=0, context="category_min[%s]" % cat,
        )
    return normalized


def _loc_day_key(item) -> Optional[tuple]:
    """
    (локация, день) из cluster_key или None, если item не попадает
    под location-cap (нет cluster_key или корзина 'other').

    Парсинг ВЫРОВНЕН с _cap_by_location (nit, wave 4): cluster_key без
    дня ('loc|x') даёт ('loc', 'unknown') — ровно ту группу, по которой
    cap режет item. Раньше _loc_day_key возвращал для такого item None,
    и гарантия не защищала его, хотя cap его кэпил (в ('loc','unknown'))
    → гарантийный item молча отсекался. None — только там, где cap
    действительно не применяется.
    """
    cluster_key = getattr(item, 'cluster_key', None)
    if not cluster_key:
        return None
    parts = cluster_key.split('|')
    location = parts[0]
    if location == 'other':
        return None
    pub_day = parts[2] if len(parts) > 2 else 'unknown'
    return (location, pub_day)


def _parse_tags(tags_str: Optional[str]) -> List[str]:
    """Парсит JSON-массив тегов из БД."""
    if not tags_str:
        return []
    try:
        import json
        return json.loads(tags_str)
    except Exception:
        return []

# Словарь алиасов source → каноническое имя (ключи ДОЛЖНЫ быть в
# нормализованном виде: lowercase без пунктуации/пробелов — см. _infer_category).
# Все пробелы/апострофы удаляются: "Tom's Hardware"→"tomshardware",
# "Krebs On Security"→"krebsonsecurity", "nasa spaceflight"→"nasaspaceflight".
_SOURCE_ALIASES = {
    "tomshardware": "tomshardware",
    "krebsonsecurity": "krebs",
    "nasaspaceflight": "nasaspaceflight",
    "habr": "habr",
    "habrnews": "habr",
    "statnews": "statnews",
    "stat": "statnews",
    "sciencedaily": "sciencedaily",
    "physorg": "physorg",
    "bleepingcomputer": "bleepingcomputer",
    "n1": "n1",
    "nplus1": "n1",
    # P2-фикс: нормализация убирает ПРОБЕЛЫ, поэтому "Naked Science"
    # → "nakedscience". Прежний ключ 'naked science' не совпадал никогда.
    "nakedscience": "nakedscience",
}


def _infer_category(item) -> str:
    """
    Определяет категорию новости по source + tags.
    categories: cinema, games, space, tech, cyber, science

    source нормализуется: lowercase + убирается пунктуация/апострофы.
    Словарь алиасов → канонический ключ (см. _SOURCE_ALIASES).
    """
    raw = (item.source or "").lower()
    import re
    # Убираем всё что не буква/цифра (включая Unicode-апострофы, тире, пробелы)
    # "Tom's Hardware" → "tomshardware"; "Krebs On Security" → "krebsonsecurity"
    normalized = re.sub(r"[^\w]", "", raw, flags=re.UNICODE)

    source = _SOURCE_ALIASES.get(normalized, normalized)

    tags = _parse_tags(getattr(item, 'tags', None))
    tags_lower = [t.lower() for t in tags]

    # Кино
    if source in ('deadline', 'variety'):
        return 'cinema'
    if any(t in tags_lower for t in ('film', 'movie', 'box office', 'reviews',
                                      'oscars', 'cannes', 'streaming', 'tv', 'broadway')):
        return 'cinema'

    # Игры
    if source in ('ign', 'eurogamer', 'dtf'):
        return 'games'
    if any(t in tags_lower for t in ('games', 'gaming', 'playstation', 'xbox', 'nintendo', 'steam')):
        return 'games'

    # Космос
    if source in ('spacenews', 'nasa', 'nasaspaceflight'):
        return 'space'
    if any(t in tags_lower for t in ('space', 'nasa', 'rocket', 'satellite', 'astronaut')):
        return 'space'

    # Тех
    if source in ('tomshardware', 'habr', 'techcrunch'):
        return 'tech'
    if any(t in tags_lower for t in ('technology', 'ai', 'tech', 'software', 'hardware')):
        return 'tech'

    # Кибербезопасность
    if source in ('krebs', 'bleepingcomputer'):
        return 'cyber'
    if any(t in tags_lower for t in ('cyber', 'security', 'malware', 'hack', 'vulnerability')):
        return 'cyber'

    # Наука
    if source in ('sciencedaily', 'physorg', 'n1', 'nakedscience', 'statnews'):
        return 'science'
    if any(t in tags_lower for t in ('science', 'research', 'study', 'discovery', 'medical', 'health')):
        return 'science'

    return 'other'


def _cap_by_location(
    items: List[NewsItem],
    max_per_location: int = 2,
    protected_urls: Optional[Set[str]] = None,
) -> List[NewsItem]:
    """
    Ограничивает число новостей по локации и дню.

    Для каждой пары (локация, день публикации) оставляет не более
    max_per_location новостей. Items с cluster_key=None или корзиной
    'other' пропускаются (кэп на них не распространяется — MAJOR 4-фикс:
    'other' не ограничивается локацией и не считается в pre_cap_loc_day).

    Алгоритм: итерация по порядку, для каждой (локация, день) сохраняются
    первые max_per_location items — остальные отфильтровываются.
    Вызывается в run_digest ПОСЛЕ _category_guarantee, ДО backfill.

    P2-фикс: protected_urls — items, закрывающие минимум по категориям
    (до category_min на категорию; см. run_digest). Иначе гарантия
    бессмысленна: guarantee-items дописываются в конец списка, а cap идёт
    по порядку → они первыми попадают под отсечку, если (location, day)
    уже набрана из топов. Protected items обрабатываются ПЕРВЫМИ
    (считаются в квоту) и имеют приоритет перед обычными.

    Minor 2-фикс (wave 2): защита НЕ может превышать сам cap — на
    (location, day) остаётся не более max_per_location items, включая
    protected. Иначе при sum(category_min) ≈ max_items protected-ы
    вытесняли всю обычную выборку и cap переставал работать молча.
    Какие именно protected-ы выживут — решает вызывающая сторона
    (run_digest обрезает набор заранее, с warning по категориям); здесь
    лишние просто отсекаются как hard backstop.
    """
    if max_per_location <= 0 or not items:
        return list(items)

    protected = protected_urls or set()
    location_day_counts: Dict[tuple, int] = {}
    result: List[NewsItem] = []
    protected_dropped = 0

    # Protected — первыми: они занимают квоту и отсекаются последними.
    ordered_items = (
        [it for it in items if getattr(it, 'url', None) in protected]
        + [it for it in items if getattr(it, 'url', None) not in protected]
    )

    for item in ordered_items:
        cluster_key = getattr(item, 'cluster_key', None)
        if not cluster_key:
            result.append(item)
            continue

        parts = cluster_key.split('|')
        location = parts[0] if parts else 'other'
        pub_day = parts[2] if len(parts) > 2 else 'unknown'

        # bucket 'other' is not capped
        if location == 'other':
            result.append(item)
            continue

        key = (location, pub_day)
        count = location_day_counts.get(key, 0)
        if count >= max_per_location:
            # Квота группы исчерпана; protected здесь тоже не проходит —
            # защита не может превышать cap.
            if getattr(item, 'url', None) in protected:
                protected_dropped += 1
            continue
        result.append(item)
        location_day_counts[key] = count + 1

    if protected_dropped:
        logger.warning(
            "Location cap: отсечено %d protected items — защита категорий "
            "превышает location-cap", protected_dropped,
        )

    if len(result) < len(items):
        locations_hit = len(location_day_counts)
        logger.info(
            "Location cap: removed %d items across %d locations",
            len(items) - len(result), locations_hit,
        )

    return result

def _category_guarantee(
    items: List[NewsItem],
    category_min: Dict[str, int],
    pool: Optional[List[NewsItem]] = None,
) -> List[NewsItem]:
    """
    Гарантирует минимум новостей по категориям.

    Алгоритм:
    1. Считаем текущее покрытие по категориям в items.
    2. Для каждой категории в category_min (кино первый): если покрытие < min,
       вытесняем лучшие по score items категории из pool,
       заменяя ими наименее ценные EVICTABLE items текущего набора.
       EVICTABLE = item, чья категория НЕ в category_min
       ИЛИ cat_counts[c] > category_min[c] (покрыта с запасом).
    3. После каждого swap инкрементально обновляем cat_counts.
    4. При пустом pool или None — legacy-режим: ничего не меняется
       (swap невозможен, гарантия остаётся no-op).

    category_min нормализуется в Dict[str, int] (_normalize_category_min):
    строковые значения из YAML не должны давать TypeError в сравнении
    (Minor 1-фикс, wave 2).

    Сорт вывода — по llm_score DESC вызывающей стороной после вызова.
    """
    category_min = _normalize_category_min(category_min)
    if not category_min or not items:
        return items

    result = list(items)
    used_ids: Set[int] = {id(it) for it in items}

    # Считаем текущее покрытие по категориям
    cat_counts: Dict[str, int] = {}
    for item in result:
        cat = _infer_category(item)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # Pool сортирован по score DESC
    pool_sorted: List[NewsItem] = []
    if pool:
        pool_sorted = sorted(
            list(pool),
            key=lambda x: x.llm_score or 0,
            reverse=True,
        )

    # Приоритет по порядку в category_min (кино первый)
    for cat, min_count in category_min.items():
        current = cat_counts.get(cat, 0)
        if current >= min_count:
            continue  # уже достаточно

        needed = min_count - current
        added = 0
        pool_idx = 0
        while added < needed and pool_idx < len(pool_sorted):
            candidate = pool_sorted[pool_idx]
            pool_idx += 1
            if id(candidate) in used_ids:
                continue  # уже в result
            if _infer_category(candidate) != cat:
                continue  # не из нужной категории

            # Жертва — только evictable: категория НЕ в category_min
            # ИЛИ cat_counts[c] > category_min[c] (покрыта с запасом).
            # Сортируем evictable по score ASC -> жертва с минимальным score.
            def _evictable(x: NewsItem) -> bool:
                c = _infer_category(x)
                return (c not in category_min) or (cat_counts.get(c, 0) > category_min[c])

            evictable = [x for x in result if _evictable(x)]
            if not evictable:
                logger.warning(
                    "Категория %s: минимум не достигнут, некого вытеснять "
                    "(current=%d, min=%d)", cat, current, min_count,
                )
                break

            evictable_sorted = sorted(evictable, key=lambda x: x.llm_score or 0)
            victim = evictable_sorted[0]  # наименее ценный evictable

            victim_cat = _infer_category(victim)
            # Удаление по индексу (вместо result.remove, опирающегося на __eq__)
            victim_idx = next(i for i, x in enumerate(result) if x is victim)
            del result[victim_idx]
            used_ids.discard(id(victim))

            # Инкрементальное обновление cat_counts
            cat_counts[victim_cat] = cat_counts.get(victim_cat, 0) - 1

            result.append(candidate)
            used_ids.add(id(candidate))
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            added += 1

    # Финальный warning по невыполненным категориям
    for cat, min_count in category_min.items():
        current = cat_counts.get(cat, 0)
        if current < min_count:
            logger.warning(
                "Категория %s: минимум не достигнут (current=%d, min=%d)",
                cat, current, min_count,
            )

    return result


def _iso_dt(value: str):
    """Парсит дату из различных форматов в datetime (UTC или timezone-aware).
    Поддерживает:
      - ISO-8601 (в т.ч. 'Z', 'YYYY-MM-DD HH:MM')
      - Русский формат 'DD.MM.YYYY HH:MM[:SS]' (без тайзоны, считается UTC)
      - Русский формат с суффиксом 'UTC': 'DD.MM.YYYY HH:MM UTC'
      - RFC-2822: 'Mon, 01 Jan 2025 12:00:00 GMT'
    Форматы без тайзоны считаются UTC.
    None при ошибке."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        pass
    # Русские форматы БЕЗ суффикса UTC
    for fmt in ("%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S"):
        try:
            return datetime.strptime(v, fmt).replace(tzinfo=tz.utc)
        except ValueError:
            continue
    # Русские форматы С суффиксом 'UTC' (основной прод-формат, collector.py)
    # Nit 1: точный срез вместо rsplit — rsplit(" UTC") срезал бы и " UTC+3".
    # Проверяем точное окончание; HH:MM и HH:MM:SS различаем перебором форматов.
    if v.endswith(" UTC"):
        base = v[:-4]  # срез " UTC" (4 символа)
        for fmt in ("%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S"):
            try:
                return datetime.strptime(base, fmt).replace(tzinfo=tz.utc)
            except ValueError:
                continue
    # RFC-2822 fallback (collector.py сырые значения типа "Mon, 01 Jan 2025 12:00:00 GMT")
    try:
        parsed = _parsedate_to_dt(v)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=tz.utc)
        return parsed
    except (TypeError, ValueError):
        pass
    return None


class NewsPipeline:
    def __init__(self, config: dict, shutdown_event=None):
        """
        Parameters
        ----------
        config : dict
            Полный конфиг приложения.
        shutdown_event : threading.Event, optional
            Событие graceful shutdown. Пробрасывается в TelegramDelivery,
            чтобы его 429/pacing-ожидания прерывались при остановке
            (иначе stop_grace_period истекает → SIGKILL).
            None → прежнее поведение (ожидания не прерываются).
        """
        self.config = config
        self.sources = config.get("sources", []) or []
        self._shutdown_event = shutdown_event

        db_path = self._resolve_db_path(config)
        dedup_cfg = config.get("dedup", {})
        self.storage = Storage(db_path, dedup_config=dedup_cfg)

        self.delivery: Delivery = self._build_delivery(config)

        # LLM-фильтр: только если включён И есть api_key
        self.llm_filter: Optional[LLMFilter] = self._build_llm_filter(config)

        # лимит items (оценок) на цикл
        self._max_per_cycle = int(config.get("llm", {}).get("max_per_cycle", 150))
        self._llm_calls_left: int = self._max_per_cycle  # общий бюджет на цикл
        # off-peak гейт: ограничить LLM вызовами в непиковые часы DeepSeek
        # явный парсер: bool("false")==True, поэтому проверяем строковые значения
        off_peak_raw = config.get("llm", {}).get("off_peak_only", False)
        self._off_peak_only = _parse_bool(off_peak_raw)
        # лимит retry-попыток
        self._max_retries = int(
            config.get("database", {}).get("max_retries", Storage.DEFAULT_MAX_RETRIES)
        )

        # кеш фидбэка для few-shot (Фаза 2c); fetched один раз за цикл
        self._feedback_context: Optional[tuple] = None

        # URLs, помеченные LLM-дедупом как дубли (сбрасываются каждый цикл)
        self._llm_dedup_skipped_urls: Set[str] = set()

    def _resolve_db_path(self, config: dict) -> str:
        """
        Резолвит database.path от единого проектного корня (config._find_project_root).
        Гарантирует существование и записываемость директории.
        Не зависит от CWD.
        """
        from . import config as cfg_module

        root = cfg_module._find_project_root()
        rel = config.get("database", {}).get("path", "data/newsbot.db")
        path = root / rel
        # Гарантируем существование и записываемость директории.
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _retry_threshold(self) -> int:
        """Порог retry в секундах из конфига или default."""
        return self.config.get("database", {}).get(
            "retry_threshold_sec",
            Storage.DEFAULT_RETRY_THRESHOLD_SECONDS,
        )

    def _is_paused(self) -> bool:
        """Проверяет флаг паузы в meta (ключ 'paused')."""
        try:
            val = self.storage.get_meta("paused")
            return val == "1"
        except Exception as e:
            logger.warning("_is_paused: %s", e)
            return False

    def _cleanup(self) -> None:
        """
        Очистка: purge_seen_titles + purge_items по retention из конфига.
        Вызывается во всех трёх ветках конвейера (top-of-cycle pause,
        off-peak, normal) — единая точка (DRY).
        """
        self.storage.purge_seen_titles()
        retention = self.config.get("database", {}).get("retention_days", 90)
        feedback_ret = self.config.get("feedback", {}).get("retention_days", 180)
        deferred_ret = self.config.get("database", {}).get(
            "deferred_retention_days",
            self.storage.DEFAULT_DEFERRED_RETENTION_DAYS,
        )
        self.storage.purge_items(
            retention_days=retention,
            feedback_retention_days=feedback_ret,
            deferred_retention_days=deferred_ret,
        )

    def _load_feedback_context(self) -> None:
        """
        Загружает последние оценки пользователя из БД для few-shot промпта.
        Вызывается ОДИН раз за цикл (перед началом фильтрации), чтобы
        передать контекст в LLMFilter.evaluate() для каждой новости.

        Гейт: feedback.enabled == True. При отключении — ничего не делает.
        Ошибки получения фидбэка — best-effort (не ломают фильтрацию).
        """
        # Minor T4-фикс: _parse_bool для консистентности с __main__.py и run_digest.
        if not _parse_bool(self.config.get("feedback", {}).get("enabled", True)):
            self._feedback_context = None
            return

        max_examples = int(
            self.config.get("feedback", {}).get("max_examples", 10)
        )

        dislikes: List[dict] = []
        likes: List[dict] = []

        try:
            dislikes = self.storage.get_recent_feedback(
                verdict="dislike", limit=max_examples
            )
        except Exception as e:
            logger.warning(
                "_load_feedback_context: ошибка получения dislikes: %s", e
            )

        try:
            likes = self.storage.get_recent_feedback(
                verdict="like", limit=max_examples
            )
        except Exception as e:
            logger.warning(
                "_load_feedback_context: ошибка получения likes: %s", e
            )

        # Оба пусты — сбрасываем в None (фильтр не вставляет блок)
        if not dislikes and not likes:
            self._feedback_context = None
        else:
            self._feedback_context = (dislikes, likes)

    def llm_dedup_enabled(self) -> bool:
        """
        Проверяет флаг llm_dedup в конфиге.
        Дефолт True, парсинг bool из конфига (безопасный, как _parse_bool).
        """
        raw = self.config.get("llm", {}).get("llm_dedup", True)
        return _parse_bool(raw)

    def _run_llm_dedup(self, candidates: List[NewsItem]) -> int:
        """
        Выполняет LLM-дедупликацию для candidates.

        Вызывается ОДИН раз за цикл, ПОСЛЕ гейтов паузы и off-peak,
        ДО скоринга/оценки. Берёт recent titles из БД (кросс-цикловый дедуп).
        Для найденных дублей:
          (a) устанавливает state='skipped' через set_state (для дублей, которые
              УЖЕ в БД со state='new' — корректная перезапись; не спамит новыми
              INSERT при повторном вызове в том же цикле)
          (б) ИСКЛЮЧАЕТ из дальнейшей обработки (не скорится/не доставляется)

        При полном сбое LLM → 0 (безопасный пропуск: ничего не пропущено).
        При llm_dedup=false или <2 кандидатов → 0, вызов не делается.

        Бюджет (max_per_cycle):
            — Детекция использует ceil(N/40) вызовов detect_duplicates.
            — Каждый чанк списывается из _llm_calls_left (1 вызов = 1 списание).
            — Если бюджета недостаточно — дедуп пропускается (безопасный fallback).
            — Семантика: бюджет покрывает и дедуп, и скоринг; приоритет у скоринга
              (retry и свежие кандидаты важнее дедупа).

        NOTE (Minor #6): recovered-items (из прошлого цикла) НЕ проходят
        LLM-дедуп как кандидаты — это осознанно. Они уже получили
        минимальную оценку при первой обработке; повторный дедуп в рамках
        того же «события» не даст новой информации, а токены потратит.
        Кросс-цикловый дедуп относительно recent-новостей покрывает их
        опосредованно (recent включает delivered/deferred/etc. из прошлых циклов).

        Returns:
            Количество отсеянных LLM-дублей.
        """
        if not candidates:
            return 0

        # Инициализируем множество, если атрибут отсутствует (тесты через __new__)
        if not hasattr(self, '_llm_dedup_skipped_urls'):
            self._llm_dedup_skipped_urls: Set[str] = set()

        dedup_cfg = self.config.get("dedup", {})
        window_days = int(dedup_cfg.get("window_days", 3))
        recent_limit = int(dedup_cfg.get("llm_recent_limit", 20))

        # Major #3-фикс: оцениваем стоимость дедупа ДО вызова.
        # detect_duplicates → ceil(N/40) чанков = ceil(N/40) вызовов.
        # Если бюджета недостаточно — пропускаем (безопасный fallback).
        num_chunks = math.ceil(len(candidates) / self.llm_filter._DEDUP_CHUNK_SIZE)
        if self._llm_calls_left < num_chunks:
            logger.info(
                "LLM-дедуп: бюджет %d < %d (нужно %d чанков) — пропускаем, "
                "бюджет сохранён для скоринга (%d items)",
                self._llm_calls_left, num_chunks, num_chunks, len(candidates),
            )
            return 0
        # Списываем бюджет заранее (до вызова — списываемое == ceil(N/40)).
        # Детектор сам глотает LLMError и возвращает пустое множество,
        # поэтому реальных сбоев после списания почти не бывает —
        # возврат через except почти недостижим. Но guard оставлен
        # для устойчивости к непредвиденным исключениям (сбой сети и т.п.).
        self._llm_calls_left -= num_chunks

        # Собираем recent titles из БД
        recent = self.storage.get_recent_titles(
            window_days=window_days,
            limit=recent_limit,
        )
        # Преобразуем в формат, ожидаемый detect_duplicates
        recent_list = [{"title": t, "source": s} for t, s in recent]

        try:
            dup_0based = self.llm_filter.detect_duplicates(candidates, recent_list)
        except Exception as e:
            # Возврат при непредвиденном сбое (LLMError уже обработан внутри
            # detect_duplicates, но на случай непредвиденных исключений — возврат).
            self._llm_calls_left += num_chunks
            logger.warning("_run_llm_dedup: исключение: %s — пропускаем (безопасно)", e)
            return 0

        if not dup_0based:
            return 0

        # Собираем URLs дублей
        dup_urls = [candidates[idx].url for idx in sorted(dup_0based) if idx < len(candidates)]

        # Critical #2-фикс: устанавливаем state='skipped' РЕАЛЬНОЙ перезаписью.
        # Для дублей, которые УЖЕ в БД со state='new' (через save_many):
        #   save_skipped = INSERT OR IGNORE → rowcount=0 → state остаётся 'new'
        #   → K1-recovery подхватит их в след.цикле или они застрянут навсегда.
        # set_state — UPDATE, который РЕАЛЬНО меняет state='new' → 'skipped'.
        if dup_urls:
            marked = self.storage.set_state(dup_urls, "skipped")
            logger.debug(
                "LLM-дедуп: set_state('skipped') для %d дублей → %d строк обновлено",
                len(dup_urls), marked,
            )

        # (б) запоминаем для исключения из дальнейшей обработки
        self._llm_dedup_skipped_urls.update(dup_urls)

        logger.info(
            "LLM-дедуп: отсеяно %d/%d кандидатов (семантические дубли «одно событие»), "
            "бюджет списан: %d (дедуп) + далее скоринг",
            len(dup_urls), len(candidates), num_chunks,
        )
        return len(dup_urls)

    def _cluster_deferred_duplicates(self, items: List[NewsItem]) -> Set[str]:
        """
        Семантическая кластеризация deferred-новостей ПЕРЕД дайджестом.

        Финальный сейф от дублей: новости одного события, накопленные
        за несколько циклов сбора, группируются LLM; дублями считаются
        все, кроме лидера — первого в текущем порядке выдачи
        (наиболее приоритетного). Лидеры остаются в дайджесте.

        Возвращает set URL'ов-дублей. При любой ошибке / отсутствии LLM /
        выключенном флаге / пике DeepSeek — пустое множество
        (безопасно: дубли уйдут как есть, ничего не теряется).
        """
        if not items or len(items) < 2:
            return set()
        digest_cfg = self.config.get("digest", {}) or {}
        if not _parse_bool(digest_cfg.get("cluster_duplicates", True)):
            return set()
        # Cost-инвариант: в пик DeepSeek LLM не вызываем (как _run_llm_dedup).
        off_peak_only = getattr(self, "_off_peak_only", False)
        if off_peak_only and not is_llm_allowed_now(off_peak_only):
            logger.info(
                "Дайджест: кластеризация дублей пропущена (пик DeepSeek, off_peak_only)",
            )
            return set()
        llm_filter = getattr(self, "llm_filter", None)
        if llm_filter is None:
            return set()
        try:
            dup_idx = llm_filter.detect_duplicates(list(items), [])
        except Exception as e:
            logger.warning("Дайджест: кластеризация дублей упала: %s", e)
            return set()
        if not isinstance(dup_idx, (set, frozenset)):
            return set()
        urls: Set[str] = set()
        for i in dup_idx:
            # Индекс 0 — лидер первой чанки: он не может быть дублем.
            if i == 0:
                continue
            if isinstance(i, int) and 0 <= i < len(items):
                urls.add(items[i].url)
        # Инвариант: хотя бы один item (лидер) должен остаться в дайджесте.
        if len(urls) >= len(items):
            logger.warning(
                "Дайджест: кластеризация пометила все %d items дублями — "
                "игнорируем результат (лидер должен остаться)", len(items),
            )
            return set()
        if urls:
            logger.info(
                "Дайджест: кластеризация отсеяла %d дублей из %d",
                len(urls), len(items),
            )
        return urls

    def _build_delivery(self, config: dict) -> Delivery:
        token = config.get("telegram", {}).get("token", "") or ""
        chat_id = config.get("telegram", {}).get("chat_id", "") or ""

        if token and chat_id:
            logger.info("Telegram token обнаружен — TelegramDelivery")
            return TelegramDelivery(
                token=token,
                chat_id=chat_id,
                shutdown_event=self._shutdown_event,
            )
        else:
            logger.info("Telegram token пуст — ConsoleDelivery")
            return ConsoleDelivery()

    def _build_llm_filter(self, config: dict) -> Optional[LLMFilter]:
        """
        Создаёт LLMFilter если llm.enabled=True и есть api_key.
        Без ключа — логирует warning, возвращает None.
        """
        llm_cfg = config.get("llm") or {}
        if not _parse_bool(llm_cfg.get("enabled", True)):
            logger.info("LLM-фильтр выключен в конфиге")
            return None

        api_key = llm_cfg.get("api_key") or ""
        if not api_key:
            logger.warning(
                "LLM отключён: нет api_key — новости будут слаться непрофильтрованными"
            )
            return None

        try:
            # storage необходим cost-guard'у (дневной лимит llm.daily_cost_limit):
            # без него _check_daily_cost всегда возвращает True.
            flt = LLMFilter(llm_cfg, storage=self.storage)
            mode = "urgent-only" if flt.push_urgent_only else "legacy"
            logger.info("LLM-фильтр инициализирован (%s / th=%d / mode=%s)",
                         flt.model, flt.push_threshold, mode)
            return flt
        except Exception as e:
            logger.error("Не удалось инициализировать LLM-фильтр: %s", e)
            logger.warning("LLM отключён: новости будут слаться непрофильтрованными")
            return None

    def run_cycle_with_delivery(self) -> None:
        """
        Один цикл сбора + обработки.

        at-least-once семантика:
          1. new-items записываются с state='new' (не pending).
          2. new-items ПОДХВАТЫВАЮТСЯ в начале цикла (recovery).
             Если процесс упал после сохранения new, но до перевода в pending —
             они будут обработаны в следующем цикле (at-least-once ✅).
          3. state переводится в pending перед set_state + доставкой.
          4. delivered выставляется после успешного send_batch.
             ⚠️ ОКНО КРЭША: между return из send_batch и set_state('delivered').
             При падении в этом окне item остаётся 'pending', retry доставит его
             повторно (возможен дубль на стороне TG). Это допустимый риск
             at-least-once; дедуп на стороне TG не реализуется.
        """
        logger.info("=== Начало цикла ===")

        # сброс usage-аккумулятора на начало цикла
        if self.llm_filter is not None:
            client = getattr(self.llm_filter, "client", None)
            if client is not None:
                usage_stats = getattr(client, "usage_stats", None)
                if usage_stats is not None:
                    usage_stats.reset()

        collector = FeedCollector(self.sources)

        # ── K1: подхватываем 'new'-items из прошлого цикла (recovery) ──
        # 2b-фикс: лимит на чтение, чтобы не грузить неограниченный объём в RAM
        # при долгой паузе. Остаток подхватится в следующих циклах (at-least-once).
        recovered: List[NewsItem] = []
        try:
            recovery_limit = int(
                self.config.get("database", {}).get("recovery_new_limit", 500)
            )
            recovered = self.storage.get_all_by_state("new", limit=recovery_limit)
            if recovered:
                total_new_count = self.storage.count_by_state("new")
                logger.info(
                    "K1-recovery: подхвачено %d/%d items со state='new' (лимит=%d)",
                    len(recovered), total_new_count, recovery_limit,
                )
                if total_new_count > recovery_limit:
                    logger.info(
                        "K1: остаток %d items со state='new' ждёт следующего цикла",
                        total_new_count - recovery_limit,
                    )
        except Exception as e:
            logger.warning("K1-recovery: ошибка чтения 'new'-items: %s", e)

        total_collected = 0
        total_new = 0
        total_dups = 0
        failed_feeds: List[str] = []
        attempted_feeds = 0
        candidates: List[NewsItem] = []

        # ── Сбор фидов ──
        try:
            for source_name, result in collector.collect():
                if not result.ok:
                    failed_feeds.append(source_name)
                    continue

                attempted_feeds += 1
                items = result.items
                total_collected += len(items)

                dedup_cfg = self.config.get("dedup", {})
                new_items, _, dup_count = filter_new(items, self.storage, dedup_config=dedup_cfg)
                total_dups += dup_count

                # save_many — батчевая вставка (В5); N1 возвращает реально вставленные NewsItem
                new_per_source = 0
                if new_items:
                    inserted_items = self.storage.save_many(new_items)
                    new_per_source = len(inserted_items)
                    total_new += new_per_source
                    candidates.extend(inserted_items)

                logger.info(
                    "[%s] собрано=%d новых=%d дублей=%d",
                    source_name, len(items), new_per_source, dup_count,
                )
        finally:
            # K3: collector.close() ВСЕГДА — даже при исключении.
            collector.close()

        logger.info(
            "=== Итого: собрано=%d новых=%d дублей=%d | фидов: OK=%d failed=%d ===",
            total_collected, total_new, total_dups,
            attempted_feeds, len(failed_feeds),
        )
        if failed_feeds:
            logger.warning("Недоступные фиды: %s", ", ".join(failed_feeds))

        # ── подготовка единого списка items на обработку ──
        # recovered (K1) + candidates (сбор)
        all_items: List[NewsItem] = recovered + candidates
        # обнуляем бюджет на начало доставки
        self._llm_calls_left = self._max_per_cycle

        # ── Few-shot feedback context (Фаза 2c): один раз за цикл ──
        self._load_feedback_context()

        # ── C2: recovery — терминализируем зависшие 'pending' с retry_count>=max ──
        # M3: retry_threshold_sec из конфига (не хардкод), чтобы совпадал с get_stuck_pending
        threshold = self._retry_threshold()
        self.storage.recover_stuck_pending(self._max_retries, retry_threshold_sec=threshold)

        # ── off-peak гейт: проверяем ДО любого вызова LLM ──
        # Paused-гейт ВЫШЕ off-peak: пауза приоритетнее off-peak.
        # При паузе: сбор/дедуп/сохранение уже выполнены (items в state='new'),
        # останавливаемся до LLM/доставки/дайджеста/алертов.
        if self._is_paused():
            total_new_items_count = self.storage.count_by_state("new")
            pending_items = self.storage.get_stuck_pending(
                max_age_seconds=threshold,
                max_retries=self._max_retries,
            )
            total_waiting = total_new_items_count + len(pending_items)
            logger.info(
                "PAUSED gate: обработка пропущена, %d items ждут возобновления "
                "(new=%d, pending=%d)",
                total_waiting, total_new_items_count, len(pending_items),
            )
            # очистка ВЫПОЛНЯЕТСЯ (purge не зависит от LLM/доставки)
            self._cleanup()
            return  # ← выходим: без LLM, доставки, дайджеста, алертов

        if self._off_peak_only and not is_llm_allowed_now(self._off_peak_only):
            # вычисляем реальное число ожидающих items до return
            stuck_items = self.storage.get_stuck_pending(
                max_age_seconds=threshold,
                max_retries=self._max_retries,
            )
            total_waiting = len(all_items) + len(stuck_items)
            logger.info(
                "off-peak gate: сейчас пик DeepSeek — LLM-обработка пропущена, "
                "%d items ждут off-peak",
                total_waiting,
            )
            # items остаются в state='new'/'pending', будут обработаны в off-peak
            # Major #1: перепроверяем паузу — она могла наступить ПОСЛЕ off-peak-гейта
            # баланс-чек вызывается и в пик — он не тратит LLM-токены (отдельный API)
            # Major #1: единая точка входа для обоих вызовов — не дублируем проверку
            self._run_delivery_side_effects()
            self._cleanup()
            return  # ← выходим из цикла; LLM не вызывается

        # ── LLM-дедупликация: семантическая группировка «одно событие» ──
        # Critical #1-фикс: вызов ПОСЛЕ гейтов паузы и off-peak.
        # На паузе и в off-peak LLM-дедуп НЕ вызывается — экономия платных токенов.
        # Кандидаты уже сохранены в БД (state='new'), лексический дедуп и seen_titles
        # также уже выполнены (до гейтов) — сохранность данных обеспечена.
        # При сбое LLM — безопасный пропуск: ни один дубль не проходит.
        llm_dedup_skipped = 0
        if self.llm_dedup_enabled() and self.llm_filter is not None:
            llm_dedup_skipped = self._run_llm_dedup(candidates)
            # Исключаем LLM-дубли из дальнейшей обработки
            if llm_dedup_skipped > 0:
                skipped_urls = getattr(self, '_llm_dedup_skipped_urls', set())
                all_items = [i for i in all_items if i.url not in skipped_urls]
            if getattr(self, '_llm_dedup_skipped_urls', None):
                # Очищаем после использования
                self._llm_dedup_skipped_urls.clear()

        # ── B13: retry-очередь обрабатывается ПЕРЕД кандидатами ──
        # M1-фикс: scored items (llm_score уже сохранён) НЕ тратят LLM-бюджет.
        #   — Разделяем stuck на scored (retry без LLM) и fresh (требуют оценки).
        #   — Scored обрабатываются БЕЗ ограничения бюджетом.
        #   — Fresh обрабатываются в рамках retry_budget (как и раньше).
        # Это решает проблему «голодовки» при _llm_calls_left==0:
        #   при исчерпанном бюджете scored-pending всё равно доставляются.
        stuck = self.storage.get_stuck_pending(
            max_age_seconds=threshold,
            max_retries=self._max_retries,
        )

        if stuck:
            # Разделяем scored vs fresh для справедливого учёта бюджета
            stuck_scored = [item for item in stuck if item.llm_score is not None]
            stuck_fresh = [item for item in stuck if item.llm_score is None]

            # Scored: обрабатываются БЕЗ ограничения LLM-бюджетом
            # Major-фикс: budget_override=len(stuck_scored) — явно резервируем
            # бюджет для scored-items. При _llm_calls_left==0 бюджет всё равно
            # расходуется только fresh-items (retry_batch пропускает LLM), scored
            # доставляются без среза.
            if stuck_scored:
                logger.info(
                    "B13: retry queue %d items (%d scored, %d fresh), "
                    "LLM-бюджет=%d/%d",
                    len(stuck), len(stuck_scored), len(stuck_fresh),
                    self._llm_calls_left, self._max_per_cycle,
                )
                failed_retry_scored = self._process_batch(
                    stuck_scored, budget_override=len(stuck_scored)
                )
                if failed_retry_scored:
                    self.storage.increment_retry_count(failed_retry_scored)
                    self._check_and_mark_failed(failed_retry_scored)
            else:
                failed_retry_scored = []

            # Fresh retry: бюджет по-старому (минимум половины или len, кламп)
            retry_budget: int
            if stuck_fresh:
                retry_budget = max(self._llm_calls_left // 2, len(stuck_fresh))
                retry_budget = min(retry_budget, self._llm_calls_left)
                if stuck_scored:
                    logger.info(
                        "B13: retry fresh %d items — бюджет %d/%d",
                        len(stuck_fresh), retry_budget, self._llm_calls_left,
                    )
                failed_retry_fresh = self._process_batch(stuck_fresh, budget_override=retry_budget)
                if failed_retry_fresh:
                    self.storage.increment_retry_count(failed_retry_fresh)
                    self._check_and_mark_failed(failed_retry_fresh)
                # Объединяем невыполненные scored + fresh
                failed_retry_urls = failed_retry_scored + failed_retry_fresh
            else:
                failed_retry_urls = failed_retry_scored
        else:
            retry_budget = 0

        # ── доставка новых candidates + recovered ──
        remaining_budget = self._llm_calls_left
        if all_items:
            logger.info(
                "Доставка %d candidates+recovered, бюджет %d/%d",
                len(all_items), remaining_budget, self._max_per_cycle,
            )
            # N1: candidates теперь реально вставленные items
            # 2c-фикс: remainder кандидатов (не доставлены по сбой TG API)
            # инкрементируют retry_count и терминализируются при исчерпании лимита.
            # Это распространяет at-least-once на candidates, а не только на retry-очередь.
            failed_candidate_urls = self._process_batch(
                all_items, budget_override=remaining_budget
            )
            if failed_candidate_urls:
                self.storage.increment_retry_count(failed_candidate_urls)
                self._check_and_mark_failed(failed_candidate_urls)

        # ── единое логирование LLM-usage за весь цикл (после обоих батчей) ──
        self._log_llm_usage()

        # Major #1: единая точка — перепроверяем паузу после батчей.
        # При паузе пропускаем дайджест и баланс-алерт, но purge выполняется
        # в вызывающем коде (выше).
        self._run_delivery_side_effects()

        # ── очистка ──
        self._cleanup()

    def _process_batch(
        self,
        items: List[NewsItem],
        budget_override: Optional[int] = None,
    ) -> List[str]:
        """
        new/pending → pending → LLM-фильтр (если активен) → route →
        skipped / deferred / delivered.

        Returns:
            Список URL, которые ОСТАЛИСЬ невыполненными (не доставлены,
            не отфильтрованы в skipped/deferred) — только для них имеет
            смысл инкрементировать retry_count и проверять лимит.

        Args:
            items: список NewsItem для обработки
            budget_override: если указано — использовать этот бюджет вместо
                             остатка общего счётчика (для retry с резервацией).

        Бюджет (max_per_cycle):
            — Единица учёта: **items** (НЕ LLM-вызовы).
            — Списание: len(fresh_batch) на каждый вызов evaluate_batch
              (retry-items с llm_score бюджет не тратят).
            — При max_per_cycle=150, batch_size=10 → до 150 items за цикл.
            — При budget<=0: все items остаются pending, LLM не вызывается.
            — При budget<0: клампится в 0 (items[:0] = []).

        При ошибке доставки: items остаются 'pending', retry в следующем цикле.
        Если фильтр неактивен (нет api_key) или LLM вернул ok=False →
        всё доставляется непрофильтрованным.

        B3: retry-лимит реализован через retry_count в БД + set_failed();
            get_stuck_pending фильтрует по retry_count < лимита.
        """
        if not items:
            logger.info("Нет items для обработки.")
            return []

        urls = [i.url for i in items]
        self.storage.set_state(urls, "pending")

        # ── LLM-оценка (если фильтр активен и score ещё не сохранён) ──
        skipped_urls: List[str] = []
        deferred_urls: List[str] = []
        to_deliver: List[NewsItem] = []
        llm_batch_records: List[dict] = []

        if self.llm_filter is not None:
            push_th = self.llm_filter.push_threshold
            digest_th = self.llm_filter.digest_threshold
            push_urgent_only = self.llm_filter.push_urgent_only

            # лимит items: бюджет передан явно (retry) или остаток общего счётчика
            budget = budget_override if budget_override is not None else self._llm_calls_left
            # Minor #4-фикс: при budget<0 → клампим в 0 (items[:<0] даёт непустой префикс)
            budget = max(0, budget)
            items_to_llm = items
            if budget <= 0:
                # бюджет исчерпан — все items остаются pending.
                # C1': НЕ возвращаем как failed_urls → retry_count не инкрементируется,
                # state остаётся pending → retry в следующем цикле.
                logger.warning(
                    "LLM-бюджет цикла исчерпан (0/%d); %d items — pending (retry в след.цикле)",
                    self._max_per_cycle, len(items),
                )
                to_deliver = []  # никто не доставлен, никого не помечаем failed

            if len(items) > budget:
                logger.warning(
                    "LLM-бюджет %d/%d: обрабатываем первые %d items, "
                    "остальные %d — pending (retry в след.цикле)",
                    budget, self._max_per_cycle, budget, len(items) - budget,
                )
                items_to_llm = items[:budget]

            # batch_size из конфига; 1 = поштучный режим (всегда evaluate)
            batch_size = max(1, int(self.config.get("llm", {}).get("batch_size", 10)))

            llm_call_count = 0

            # быстрая реакция на паузу: проверяем перед каждым батчем.
            # При паузе прекращаем → оставшиеся items остаются pending (retry в след.цикле).
            for batch_start in range(0, len(items_to_llm), batch_size):
                if self._is_paused():
                    remaining = len(items_to_llm) - batch_start
                    logger.info(
                        "PAUSED mid-batch: остановлены %d unprocessed items → pending",
                        remaining,
                    )
                    break

                batch = items_to_llm[batch_start: batch_start + batch_size]

                # разделяем retry-items (llm_score есть) и fresh-items (нужна оценка)
                retry_batch = [item for item in batch if item.llm_score is not None]
                fresh_batch = [item for item in batch if item.llm_score is None]

                # ── retry: synthesize FilterResult from DB (не тратят бюджет) ──
                for item in retry_batch:
                    res = FilterResult(
                        ok=True,
                        score=item.llm_score,
                        summary=item.llm_summary or item.summary,
                        reason=item.llm_reason or "",
                        urgent=bool(item.llm_urgent) if item.llm_urgent is not None else False,
                    )
                    item._batch_result = res

                # ── fresh: батч-оценка через LLM ──
                if fresh_batch:
                    # batch_size=1 → evaluate_batch ведёт себя как evaluate
                    batch_results = self.llm_filter.evaluate_batch(
                        fresh_batch, feedback_context=self._feedback_context,
                    )
                    # Major #1-фикс: бюджет считается в items (не в вызовах).
                    # При max_per_cycle=150 и batch_size=10 → до 150 items/цикл.
                    # Списание = len(fresh_batch) — retry-items с llm_score
                    # бюджет не тратят (они уже синтезированы выше).
                    self._llm_calls_left -= len(fresh_batch)
                    llm_call_count += 1

                    # M1: собираем результаты, но НЕ применяем route() сразу —
                    # сначала вычислим cluster-max, потом применим.
                    for item, res in zip(fresh_batch, batch_results):
                        # сохраняем результат в item (всегда, даже ok=False)
                        item.llm_score = res.score
                        item.llm_summary = res.summary
                        item.llm_reason = res.reason
                        item._batch_result = res

            logger.debug("LLM-вызовов за батч: %d", llm_call_count)

            # CRITICAL #1-фикс: кластеризация ограничена items_to_llm (бюджетный
            # срез). Items за пределами бюджета остаются pending, их llm_score
            # не персистится → propagation для них не нужна.
            from collections import defaultdict
            cluster_scores: Dict[str, float] = defaultdict(lambda: 0.0)
            cluster_members: Dict[str, List[NewsItem]] = defaultdict(list)
            for item in items_to_llm:
                if getattr(item, "cluster_key", None):
                    cluster_members[item.cluster_key].append(item)
                    if item.llm_score is not None and item.llm_score > cluster_scores[item.cluster_key]:
                        cluster_scores[item.cluster_key] = float(item.llm_score)

            # CRITICAL #1-фикс: распространяем максимальный score обратно по кластеру.
            # Цикл ограничен items_to_llm (бюджетный срез не трогаем —
            # его items остаются pending без персиста). Мутируем res.score
            # ДО route(), чтобы маршрутизация и запись в llm_batch_records
            # согласованно видели повышенный score.
            for key, members in cluster_members.items():
                best_score = cluster_scores[key]
                for item in members:
                    if item.llm_score is None or item.llm_score < best_score:
                        item.llm_score = int(best_score)
                        # Мутируем FilterResult, привязанный к item.
                        # route() увидит согласованный score → правильная маршрутизация.
                        res = getattr(item, "_batch_result", None)
                        if res and res.ok:
                            res.score = int(best_score)

            # ── теперь route(): применяем результаты ко ВСЕМ items ──────────
            for item in items_to_llm:
                res = getattr(item, "_batch_result", None)
                if res is None:
                    continue
                self._apply_result(item, res, push_th, digest_th,
                                   push_urgent_only, skipped_urls, deferred_urls,
                                   to_deliver, llm_batch_records)

            # сохраняем все llm-результаты батчем (push + deferred + skipped)
            if llm_batch_records:
                self.storage.update_llm_results_batch(llm_batch_records)

            # ── фиксируем state и логируем итоговые счётчики (одна INFO-строка) ──
            n_skipped = len(skipped_urls)
            n_deferred = len(deferred_urls)
            n_push = len(to_deliver)

            if skipped_urls:
                self.storage.set_state(skipped_urls, "skipped")
            if deferred_urls:
                self.storage.set_state(deferred_urls, "deferred")

            mode_tag = " [urgent-only]" if push_urgent_only else ""
            logger.info(
                "Маршрутизация%s: push=%d deferred=%d skipped=%d",
                mode_tag, n_push, n_deferred, n_skipped,
            )

            # M1: если все отфильтрованы (to_deliver пуст) — НЕ возвращаем
            #     filtered items как failed_urls; сначала фиксируем их state.
            #     Защита: items остаются pending навсегда без этого.
            if not to_deliver:
                return []
        else:
            # фильтр неактивен — всё доставляем непрофильтрованным
            # F3: все — kind='fallback' (не путаются с push в few-shot)
            logger.info("LLM-фильтр неактивен: доставляем все %d items без фильтрации", len(items))
            to_deliver = list(items)
            for item in to_deliver:
                item._feedback_kind = "fallback"  # type: ignore[attr-defined]

        # если все отфильтрованы — нечего доставлять, возвращаем пустой remainder
        if not to_deliver:
            return []

        # ── доставка ──
        # NIT: Major #1-фикс — pause-гейт ВЫШЕ создания feedback_targets.
        # При паузе не создаём targets для недоставленных items.
        if self._is_paused():
            logger.info(
                "PAUSED перед доставкой: %d items → pending (retry в след.цикле)",
                len(to_deliver),
            )
            return []

        # F2: создаём feedback_targets ТОЛЬКО при TelegramDelivery (не Console).
        # F1: get_or_create — идемпотентно (retry не создаёт дублей).
        # F3: kind из item._feedback_kind ('push' или 'fallback').
        # best-effort: ошибка создания target НЕ ломает доставку.
        from .delivery import TelegramDelivery
        # Minor T4-фикс: _parse_bool для консистентности с __main__.py и run_digest.
        should_feedback = bool(
            _parse_bool(self.config.get("feedback", {}).get("enabled", True))
            and isinstance(self.delivery, TelegramDelivery)
        )
        if should_feedback:
            for item in to_deliver:
                kind = getattr(item, "_feedback_kind", "push")
                try:
                    item.feedback_target_id = (
                        self.storage.get_or_create_feedback_target_from_item(item, kind)
                    )
                except Exception as e:
                    logger.warning(
                        "feedback: не удалось создать target для '%s': %s",
                        (item.title or "")[:60], e,
                    )

        try:
            delivered_urls = self.delivery.send_batch(to_deliver)
            if delivered_urls:
                self.storage.set_state(delivered_urls, "delivered")
                not_delivered = [i.url for i in to_deliver
                                 if i.url not in delivered_urls]
                if not_delivered:
                    logger.info(
                        "Доставлено %d/%d items; %d не доставлено (pending → retry)",
                        len(delivered_urls), len(to_deliver), len(not_delivered),
                    )
            else:
                not_delivered = [i.url for i in to_deliver]
                logger.info("Доставлено 0 items (items остаются pending)")
        except (ChatIdNotBoundError, BotBlockedError) as e:
            # P1-фикс: ПОСТОЯННАЯ ошибка конфигурации/доступа, а не транзиентная.
            # ChatIdNotBoundError/BotBlockedError — подклассы TelegramDeliveryError,
            # поэтому раньше попадали в общий except и url'ы уходили в failed_urls →
            # increment_retry_count каждый цикл → max_retries → failed → purge.
            # Сценарий «юзер заблокировал бота на 2 часа» стирал все новости.
            # Теперь НЕ возвращаем failed: items остаются pending до починки чата,
            # retry_count не инкрементируется (по образцу digest-пути).
            logger.warning(
                "Доставка невозможна (%s): %d items остаются pending без "
                "расхода retry_count (чат/доступ не настроены)",
                type(e).__name__, len(to_deliver),
            )
            return []
        except Exception as e:
            logger.error(
                "Доставка упала (%d items → pending, retry в след.цикле): %s",
                len(to_deliver), e,
            )
            not_delivered = [i.url for i in to_deliver]

        # ── формируем множество НЕУСПЕШНЫХ URL ──
        # C1': инкремент/терминализация — ТОЛЬКО для реально не доставленных.
        # Items, срезанные бюджетом, НЕ добавляются: они остаются pending,
        # retry в следующем цикле. При llm_filter is None — всех разобрал
        # to_deliver → processed_urls = all items → excess пуст.
        processed_urls: Set[str] = (
            set(i.url for i in to_deliver) | set(skipped_urls) | set(deferred_urls)
        )
        excess_urls = [
            i.url for i in items
            if i.url not in processed_urls
        ]
        if excess_urls:
            logger.debug(
                "Бюджетный срез: %d items не получили обработки → pending (retry в след.цикле)",
                len(excess_urls),
            )
        # NOTE: excess_urls НЕ добавляются в failed_urls.

        # (a) те, что не доставлены (сбой TG, частичный батч, исключение)
        failed_urls = set(not_delivered)

        return list(failed_urls)

    def _apply_result(
        self,
        item: "NewsItem",
        res: FilterResult,
        push_th: int,
        digest_th: int,
        push_urgent_only: bool,
        skipped_urls: List[str],
        deferred_urls: List[str],
        to_deliver: List["NewsItem"],
        llm_batch_records: List[dict],
    ) -> None:
        """
        Применяет FilterResult к одному item: маршрутизация + сбор record.
        Вызывается из _process_batch после evaluate / synthesize (retry).
        """
        if not res.ok:
            if res.cost_blocked:
                # Дневной лимит LLM исчерпан: вызова не было, деньги не
                # потрачены, завтра счётчик обнулится → item НЕ трогаем:
                # остаётся 'pending' (как off-peak-гейт), не deferred/failed.
                # Warning «лимит исчерпан» уже дан один раз за день в filter.
                logger.debug(
                    "cost-blocked: '%s' остаётся pending (дневной лимит LLM)",
                    item.url,
                )
                return
            # M2-фикс: fallback при push_urgent_only=True → deferred (дайджест).
            # Это защищает от спама пушами: сбой LLM не означает срочность.
            # При push_urgent_only=False — прежнее поведение (push).
            if push_urgent_only:
                deferred_urls.append(item.url)
                llm_batch_records.append({
                    "url": item.url,
                    "score": None,
                    "summary": "",
                    "reason": "",
                    "urgent": False,
                })
            else:
                to_deliver.append(item)
                item._feedback_kind = "fallback"  # type: ignore[attr-defined]
            return

        r = route(res.score, push_th, digest_th,
                  urgent=res.urgent, push_urgent_only=push_urgent_only)
        if r == "skip":
            skipped_urls.append(item.url)
            llm_batch_records.append({
                "url": item.url,
                "score": res.score,
                "summary": res.summary,
                "reason": res.reason,
                "urgent": res.urgent,
            })
        elif r == "digest":
            deferred_urls.append(item.url)
            llm_batch_records.append({
                "url": item.url,
                "score": res.score,
                "summary": res.summary,
                "reason": res.reason,
                "urgent": res.urgent,
            })
        else:
            # r == "push"
            to_deliver.append(item)
            item._feedback_kind = "push"  # type: ignore[attr-defined]
            llm_batch_records.append({
                "url": item.url,
                "score": res.score,
                "summary": res.summary,
                "reason": res.reason,
                "urgent": res.urgent,
            })

    def _check_and_mark_failed(self, failed_retry_urls: List[str]) -> None:
        """
        C1: переводит реально невыполненные items с retry_count >= лимита
        в терминальный state='failed'. 'failed' не попадает в retry,
        удаляется purge по retention.

        M2: использует mark_failed_if_retries_exhausted (внутри — батчинг,
        фильтрация retry_count >= max_retries, WHERE url IN + AND retry_count >= ?).
        """
        if not failed_retry_urls:
            return
        try:
            marked = self.storage.mark_failed_if_retries_exhausted(
                failed_retry_urls, self._max_retries
            )
            if marked:
                logger.warning(
                    "C1: %d items превысили лимит retry (%d) → state='failed'",
                    marked, self._max_retries,
                )
        except Exception as e:
            logger.warning("_check_and_mark_failed: %s", e)

    def _log_llm_usage(self) -> None:
        """
        Логирует агрегированный LLM-usage за цикл РОВНО ОДИН раз.
        Безопасный getattr: fake-фильтры в тестах могут не иметь client/usage_stats.
        """
        if self.llm_filter is None:
            return
        stats = getattr(self.llm_filter, "client", None)
        if stats is not None:
            stats = getattr(stats, "usage_stats", None)
        if stats is not None:
            stats.log_cycle()

    def run_digest(self, force: bool = False) -> int:
        """
        Выполняет дайджест для накопленных deferred-items.
        Используется из CLI (--digest) и из демон-цикла.

        M3 (on_empty): force=True всегда выполняет digest.run_digest
        (может отправить пустой дайджест). force=False + пустой список
        → пропускает (on_empty=skip в конфиге).

        Feedback (Фаза 2b-2b): при feedback.digest_enabled=true для каждого
        отправляемого item создаётся feedback_target (kind='digest') через
        get_or_create_feedback_target_from_item (идемпотентно). Кнопки 👍/👎
        добавляются в сообщение дайджеста. Создание target — best-effort,
        ошибка не ломает дайджест. При digest_enabled=false (дефолт) —
        feedback_targets НЕ создаются, отправляется send_text (без клавиатуры).

        Args:
            force: если True — работает даже при digest.enabled=false
                   и при пустом списке deferred (для ручного --digest).

        Returns:
            Количество отправленных новостей (0 если пропущен/ошибка).
        """
        digest_cfg = self.config.get("digest", {}) or {}
        enabled = _parse_bool(digest_cfg.get("enabled", True))

        # M3: --digest (force) работает независимо от enabled
        if not enabled and not force:
            logger.debug("Дайджест выключен (digest.enabled=false)")
            return 0

        try:
            max_items = max(1, int(digest_cfg.get("max_items", 30)))
        except (TypeError, ValueError):
            max_items = 30
            logger.warning(
                "digest.max_items некорректен: %r, использую 30",
                digest_cfg.get("max_items"),
            )
        try:
            max_items_per_message = int(digest_cfg.get("max_items_per_message", 20))
        except (TypeError, ValueError):
            max_items_per_message = 20
            logger.warning(
                "digest.max_items_per_message некорректен: %r, использую 20",
                digest_cfg.get("max_items_per_message"),
            )
        on_empty = digest_cfg.get("on_empty", "skip")

        # M2: грейс published-фильтра: новости с published РАНЬШЕ window_start
        # на poll_interval выпадали навсегда. Расширяем фильтр:
        # published >= window_start - poll_interval.
        # poll_interval из конфига, default 15 мин.
        try:
            poll_interval_minutes = int(self.config.get("poll_interval_minutes", 15))
        except (TypeError, ValueError):
            poll_interval_minutes = 15  # константа: 15 мин
        grace_delta = timedelta(minutes=poll_interval_minutes)

        # ── Оконная логика: граница окна ───────────────────────────────────
        now_utc = datetime.now(tz.utc).replace(microsecond=0)
        digest_started_iso = now_utc.isoformat()
        raw_boundary = self.storage.get_meta("digest_window_started_at")
        if not raw_boundary:
            bootstrap = digest_cfg.get("bootstrap_window_hours", 24)
            try:
                bootstrap = int(bootstrap)
            except (TypeError, ValueError):
                bootstrap = 24
            window_start_iso = (now_utc - timedelta(hours=bootstrap)).isoformat()
            logger.info(
                "Дайджест: граница окна не найдена, bootstrap=%d часов → %s",
                bootstrap, window_start_iso,
            )
        else:
            window_start_iso = raw_boundary

        # Deferred items: only after window_start_iso (оконный фильтр).
        # При bootstrap (граница не из meta) фильтр НЕ применяется —
        # это защищает тесты без явной границы и первый запуск на свежей БД.
        # Следующий цикл уже имеет границу в meta и будет фильтровать.
        # Minor 3-фикс: проводка digest.max_age_hours (ранее мёртвый кноб).
        max_age_hours: Optional[int] = None
        try:
            raw_age = digest_cfg.get("max_age_hours")
            if raw_age is not None:
                max_age_hours = int(raw_age)
        except (TypeError, ValueError):
            pass
        deferred_items = self.storage.get_deferred_items(
            # M3-фикс: берём с запасом, чтобы Python-фильтр по published
            # не «съел» нужные items (SQL LIMIT отрабатывает ДО фильтра).
            # Окончательная обрезка — deferred_items[:max_items] ниже.
            limit=max_items * 2 if (max_items and max_items > 0) else None,
            since_iso=window_start_iso if raw_boundary is not None else None,
            max_age_hours=max_age_hours,
        )
        # Кросс-дайджест дедуп: items с published РАНЬШЕ начала окна не
        # попадают в дайджест (late-posted старые новости). Фильтр в Python:
        # формат published в БД неоднородный (ISO у англ. фидов,
        # 'DD.MM.YYYY HH:MM' у русских), SQLite julianday() по нему
        # возвращает NULL и отсеивал ВСЕ строки (пустые дайджесты на проде).
        # published, который не парсится, — сохраняем (не теряем новости).
        # Публикуем отфильтрованный по published список (post-published-filtered)
        # — он будет использован для бэкфилла и гарантии по категориям.
        # Сохраняем ДО обрезки: это pool за пределами top-N.
        deferrable_post_published = list(deferred_items)
        # M6: счётчики стадий
        n_sql = len(deferred_items)
        n_published = n_sql  # default: published-фильтр не отсеял ничего (bootstrap)

        if raw_boundary is not None:
            ws_dt = _iso_dt(window_start_iso)
            if ws_dt is not None:
                # M2-фикс: правильная нормализация aware→naive через astimezone,
                # а не .replace(tzinfo=None) (последнее давало ошибку ±offset часов).
                # Minor 1: если ws_dt naive (legacy-meta без offset), astimezone
                # трактовала бы его как локальное время → граница съезжает на offset.
                # Лечим: naive оставляем как есть, aware нормализуем.
                ws_naive = ws_dt if ws_dt.tzinfo is None else ws_dt.astimezone(tz.utc).replace(tzinfo=None)
                # M2: граница с учётом грейса (published может быть чуть раньше window_start)
                ws_grace_naive = ws_naive - grace_delta
                filtered = []
                for it in deferrable_post_published:
                    pub_str = str(it.published or "")
                    pub_dt = _iso_dt(pub_str) if pub_str else None
                    # Непарсящийся published сохраняем (не теряем новости).
                    # M2: aware → UTC → naive; naive уже без тайзоны.
                    if pub_dt is None:
                        filtered.append(it)
                    else:
                        if pub_dt.tzinfo is not None:
                            pub_dt_norm = pub_dt.astimezone(tz.utc).replace(tzinfo=None)
                        else:
                            pub_dt_norm = pub_dt
                        if pub_dt_norm >= ws_grace_naive:
                            filtered.append(it)
                deferrable_post_published = filtered
                n_published = len(filtered)

        # Minor 2 (NIT 1): обрезаем СРАЗУ после published-фильтра, ДО кластеризации.
        # SQL-запас *2 оставлен (страхует от того, чтобы published-фильтр не «съел»
        # нужные слоты). Truncation ВЫНЕСЕНА из if raw_boundary — применяется всегда,
        # включая bootstrap (raw_boundary=None), чтобы при отсутствии границы
        # _cluster_deferred_duplicates тоже получала ≤max_items → без 2× LLM-вызовов.
        deferred_items = (
            deferrable_post_published[:max_items]
            if (max_items and max_items > 0)
            else deferrable_post_published
        )
        n_top_n = len(deferred_items)

        logger.info(
            "Дайджест: старт — newsbot v%s, deferred=%d",
            __version__, len(deferred_items),
        )

        # M2-фикс: при пустом deferred + on_empty=skip пишем маркер ОКНА,
        # чтобы _maybe_run_digest не спамил в следующем цикле.
        # Оконная логика: пустое окно → сдвигаем границу.
        if not deferred_items and not force:
            if on_empty == "skip":
                logger.debug("Дайджест: нет отложенных новостей, окно проверено (skip)")
                self.storage.set_meta(
                    "last_digest_at",
                    digest_started_iso,
                )
                unprocessed = self.storage.get_oldest_unprocessed_fetched_at()
                new_boundary = digest_started_iso
                udt = _iso_dt(unprocessed) if isinstance(unprocessed, str) else None
                if udt is not None and udt < now_utc:
                    new_boundary = unprocessed
                self.storage.set_meta("digest_window_started_at", new_boundary)
                return 0
            # on_empty=send: обрабатывается дальше (пустое уведомление)

        # ── Финальный сейф от дублей: семантическая кластеризация deferred ──
        # Новости одного события, накопленные за несколько циклов, не должны
        # уйти в дайджест пачкой. Лидер остаётся, дубли отсеиваются.
        _dup_urls = self._cluster_deferred_duplicates(deferred_items)
        if _dup_urls:
            deferred_items = [it for it in deferred_items if it.url not in _dup_urls]
        n_dedup = len(deferred_items)

        # ── Задача A: бэкфилл после LLM-дедупа ──────────────────────────────
        # Если в топе оказались дубли — дайджест недобирается (буфер max_items..2*max_items
        # не используется). Добираем из pool_tail: items, изначально БЫВШИХ ЗА
        # пределами top-N (deferrable_post_published[max_items:]). Важно: items,
        # которые были В top-N, но были удалены dedup-ом, НЕ берутся из пула —
        # они уже прошли top-N и не являются «запасными» кандидатами.
        #
        # Бэкфилл НЕ проходит повторный LLM-дедуп: запасные items прошли
        # pre-LLM cluster_key caps при сборе и domain-дедуп — достаточно.
        # Запоминаем, какие items уже были в original top-N (до dedup).
        if deferrable_post_published and max_items and max_items > 0:
            pool_tail = deferrable_post_published[max_items:]
            backfill_added = 0
            existing_urls = {x.url for x in deferred_items}
            for it in pool_tail:
                if len(deferred_items) >= max_items:
                    break
                if it.url not in existing_urls:
                    deferred_items.append(it)
                    existing_urls.add(it.url)
                    backfill_added += 1
            if backfill_added:
                logger.info(
                    "Дайджест: бэкфилл добавил %d items из пула (top-N был неполным)",
                    backfill_added,
                )
        n_backfill = len(deferred_items)

        # ── Задача B: гарантия по категориям с возможностью ПДТЯГИВАНИЯ ──────
        # Pool для гарантии: deferrable_post_published минус всё, что уже в
        # deferred_items по url (id не используем — cap теперь после гарантии).
        # MAJOR: cap by location/day — строго ПОСЛЕ _category_guarantee.
        # Передвинут в конец, чтобы инвариант «≤max_per_location на (локация, день)»
        # держался железно даже после свапов гарантии.
        category_pool = None
        if deferrable_post_published:
            included_urls = {it.url for it in deferred_items}
            category_pool = [
                it for it in deferrable_post_published
                if it.url not in included_urls
                and it.url not in _dup_urls
            ]
        # Minor 1-фикс (wave 2): нормализация category_min — YAML может отдать
        # строку ("2"); `str <= int` и `seq[:str]` бросают TypeError ВНЕ
        # try вокруг _run_digest → дайджест падал бы каждый цикл.
        category_min = _normalize_category_min(digest_cfg.get("category_min", {}))
        # Minor 1-фикс: max_per_location вычисляется ДО блока гарантии —
        # от него зависит квота защиты (Minor 2-фикс). Невалидное значение
        # → documented default (2), а не отключение cap'а.
        raw_max = digest_cfg.get("max_per_location", 2)
        # None/"" (ключ есть, значение пустое) — это НЕ «выключить cap»,
        # а незаполненный конфиг: documented default 2 + warning. Явный 0
        # остаётся осознанным «без cap». Раньше `if raw_max` молча
        # отключал cap на пустом значении.
        if raw_max is None or (isinstance(raw_max, str) and not raw_max.strip()):
            logger.warning(
                "digest.max_per_location=%r — пустое значение, использую 2 "
                "(явный 0 отключает cap)", raw_max,
            )
            max_per_location = 2
        else:
            max_per_location = _coerce_int(
                raw_max, default=2, context="digest.max_per_location",
            )
        # P2-фикс: гарантийные items не должны отсекаться location-cap'ом.
        # Защищаем от cap ровно те items, которые закрывают минимум по
        # категории (до min_count на категорию, лучшие по score). Сюда входят
        # и items, уже бывшие в топе, и добавленные _category_guarantee из pool.
        guarantee_protected_urls: Set[str] = set()
        if category_min:
            deferred_items = _category_guarantee(deferred_items, category_min, pool=category_pool)
            _by_cat: Dict[str, List[NewsItem]] = {}
            for _it in deferred_items:
                _by_cat.setdefault(_infer_category(_it), []).append(_it)
            # Приоритет категорий — порядок в category_min (кино первый).
            _cat_priority = {c: i for i, c in enumerate(category_min)}
            # protected сгруппированы по (локация, день) — для квоты защиты.
            _protected_by_group: Dict[tuple, List[tuple]] = {}
            _protected_free: List[NewsItem] = []
            for _cat, _min_count in category_min.items():
                if _min_count <= 0:
                    continue
                _cat_items = sorted(
                    _by_cat.get(_cat, []),
                    key=lambda x: x.llm_score or 0,
                    reverse=True,
                )
                for _it in _cat_items[:_min_count]:
                    _loc_day = _loc_day_key(_it)
                    if _loc_day is None:
                        # cap на item не распространяется — защита не нужна
                        _protected_free.append(_it)
                        continue
                    _protected_by_group.setdefault(_loc_day, []).append((_cat, _it))
            for _it in _protected_free:
                guarantee_protected_urls.add(_it.url)
            # Minor 2-фикс (wave 2): защита не может превышать сам cap —
            # иначе при sum(category_min) ≈ max_items protected-ы покрывают
            # почти весь дайджест и max_per_location молча перестаёт работать.
            # При превышении оставляем категории с высшим приоритетом
            # (порядок в category_min), внутри категории — лучшие по score.
            for _loc_day, _entries in _protected_by_group.items():
                if (max_per_location and max_per_location > 0
                        and len(_entries) > max_per_location):
                    _entries.sort(
                        key=lambda ce: (
                            _cat_priority.get(ce[0], len(_cat_priority)),
                            -(ce[1].llm_score or 0),
                        )
                    )
                    _kept = _entries[:max_per_location]
                    _dropped = _entries[max_per_location:]
                    logger.warning(
                        "Дайджест: защита категорий %s превышает location-cap=%d "
                        "для локации %s/%s — оставлено %d, без защиты %s",
                        ", ".join(sorted({c for c, _ in _entries})),
                        max_per_location, _loc_day[0], _loc_day[1],
                        len(_kept),
                        ", ".join(sorted({c for c, _ in _dropped})),
                    )
                else:
                    _kept = _entries
                for _cat, _it in _kept:
                    guarantee_protected_urls.add(_it.url)
        n_guarantee = len(deferred_items)

        # MAJOR: cap by location/day — СТРОГО ПОСЛЕДНИЙ, после всех свапов.
        # Инвариант «≤max_per_location на (локация, день)» держится железно
        # (max_per_location>0 → в каждой группе ≥1), cap не может обнулить
        # список; защита категорий ограничена тем же cap'ом (Minor 2-фикс).
        capped_count = 0
        from collections import Counter
        if max_per_location and max_per_location > 0 and deferred_items:
            pre_cap: Counter = Counter()
            for it in deferred_items:
                _loc_day = _loc_day_key(it)
                if _loc_day is not None:
                    pre_cap[_loc_day] += 1
            capped_count = sum(1 for cnt in pre_cap.values() if cnt >= max_per_location)
            deferred_items = _cap_by_location(
                deferred_items, max_per_location=max_per_location,
                protected_urls=guarantee_protected_urls,
            )
            if deferred_items:
                logger.info(
                    "Дайджест: location-cap оставил %d items (%d локаций исчерпано)",
                    len(deferred_items), capped_count,
                )
        # Minor 2-фикс: n_cap фиксируется сразу после cap, ДО позднего backfill.
        n_cap = len(deferred_items)
        n_late_backfill = 0

        # M4: добор после cap. Если остались свободные слоты — добираем
        # из deferrable_post_published: items НЕ вошедшие в deferred_items
        # из-за общего top_n, но чья (локация, день) ещё не исчерпана.
        # Гарантия категорий: guarantee-items не вытесняются backfill'ом
        # (guarantee уже выполнена и гарантийные items уже в deferred_items).
        # Minor 2-фикс: комментарий уточнён — возраст уже отфильтрован в SQL
        # get_deferred_items (max_age_hours передаётся в SQL-запрос).
        backfill_added = 0
        if len(deferred_items) < max_items and deferrable_post_published:
            included_urls = {it.url for it in deferred_items}
            # MAJOR 3-фикс (wave 2): единый проход отбора — проверка и инкремент
            # running_counts В ОДНОМ цикле, где принимается решение об append.
            # Старый баг: backfill_candidates наполнялся ПОЛНЫМ проходом
            # (filter-only), а инкремент был в ОТДЕЛЬНОМ цикле ПОСЛЕ —
            # все кандидаты одной (loc,day) проходили одновременно,
            # превышая max_per_location.
            # running_counts инициализируется из post-cap deferred_items
            # (deferred_items уже прошли cap → реальные числа в дайджесте).
            # Counter импортирован выше (Minor 3-фикс).
            running_counts: Counter = Counter()
            if max_per_location and max_per_location > 0:
                for it in deferred_items:
                    _ld = _loc_day_key(it)
                    if _ld is not None:
                        running_counts[_ld] += 1
            backfill_candidates: List[NewsItem] = []
            for it in deferrable_post_published:
                if it.url in included_urls:
                    continue
                # M1: пропускаем LLM-дубли (не были в deferred_items после dedup,
                # но они в deferrable_post_published)
                if it.url in _dup_urls:
                    continue
                # Единый проход: проверка + инкремент сразу, до следующего кандидата.
                # 'other' или отсутствие cluster_key не ограничивается локацией.
                # Minor 1-фикс: guard max_per_location > 0 — без него 0>=0=True,
                # backfill отбрасывает всех (тихая деградация при max_per_location=0).
                # Nit (wave 4): группировка через _loc_day_key — как в cap и в
                # гарантии; локальный `len(parts) >= 3` иначе не считал
                # truncated-ключи и backfill возвращал отсечённый cap'ом item.
                _ld = _loc_day_key(it)
                if _ld is not None and max_per_location > 0:
                    if running_counts[_ld] >= max_per_location:
                        continue
                    running_counts[_ld] += 1
                backfill_candidates.append(it)
            # Сортировка: score DESC (fetched_at ASC уже в исходном порядке из SQL)
            backfill_candidates.sort(key=lambda x: -(x.llm_score or 0))
            for it in backfill_candidates:
                if len(deferred_items) >= max_items:
                    break
                deferred_items.append(it)
                n_late_backfill += 1
            if n_late_backfill:
                logger.info(
                    "Дайджест: backfill после cap добавил %d items "
                    "(loc free slots used)",
                    n_late_backfill,
                )
        # Minor 2-фикс: n_backfill_total = после ВСЕГО добора (ранний + поздний).
        # Ранний backfill считал items_added; поздний — n_late_backfill.
        n_backfill_total = n_backfill + n_late_backfill

        # Финальный сорт — по llm_score DESC (состав уже зафиксирован гарантийными
        # свапами; сорт только порядок вывода). Финальный slice ПОСЛЕ гарантии.
        deferred_items.sort(key=lambda x: x.llm_score or 0, reverse=True)
        deferred_items = deferred_items[:max_items]
        n_final = len(deferred_items)

        # M6: единый INFO-лог стадий (воронка: sql → published → top_n → dedup →
        #      early_backfill → guarantee → cap → late_backfill → final).
        # Minor 2-фикс: backfill_total = ранний + поздний добор; cap фиксируется
        # после cap, до позднего backfill.
        logger.info(
            "Дайджест stages: sql=%d published=%d top_n=%d dedup=%d "
            "early_backfill=%d guarantee=%d cap=%d late_backfill=%d final=%d",
            n_sql, n_published, n_top_n, n_dedup, n_backfill,
            n_guarantee, n_cap, n_late_backfill, n_final,
        )

        # ── Feedback targets + передача готовых groups в digest ──
        # F2: только при TelegramDelivery + feedback.enabled + digest_enabled.
        # digest_enabled=false (дефолт): не создаём targets (экономия БД-записей),
        # item_buttons_per_msg=None → send_text (без клавиатуры).
        # m2-фикс: единый источник — groups строятся ОДИН раз в pipeline.
        # Critical-фикс: deferred НЕ терминализируются по неуспешным попыткам
        # дайджеста. Retry_count для deferred не инкрементируется в цикле
        # дайджеста. Deferred-items остаются в 'deferred' до успешной
        # отправки (at-least-once) или штатного retention-purge.
        # NOTE: retention-purge для 'deferred' реализован через
        # deferred_retention_days в purge_items (deferred удаляются по age).
        # При транзиентном сбое отправки deferred-items остаются в очереди
        # и повторяются в следующем окне.

        # передаются в run_digest() (без пересчёта).
        # Порядок кнопок СТРОГО соответствует порядку новостей в groups.
        feedback_cfg = self.config.get("feedback", {})
        should_feedback = (
            _parse_bool(feedback_cfg.get("enabled", True))
            and _parse_bool(feedback_cfg.get("digest_enabled", False))
            and isinstance(self.delivery, TelegramDelivery)
        )

        # CRITICAL-фикс: инициализация ДО использования.
        # Раньше: `failed_digest_urls` объявлялась ПОСЛЕ build_digest_messages —
        # при исключении в рендеринге → UnboundLocalError → весь дайджест падал,
        # set_state('failed') не вызывался, маркеры окна не двигались.
        failed_digest_urls: List[str] = []

        groups, _, _build_failed = build_digest_messages(
            deferred_items,
            max_items_per_message=max_items_per_message,
        )

        # MAJOR 2-фикс: _build_failed — items, выпавшие при рендеринге
        # (уже исключены из groups). Добавляем к feedback-failed для set_state('failed').
        if _build_failed:
            for url in _build_failed:
                if url not in failed_digest_urls:
                    failed_digest_urls.append(url)

        # Фикс: оригинальное условие, но с правильным приоритетом скобок.
        # Оригинальное: not groups and not (not deferred_items and on_empty == "send")
        #   = not groups and (deferred_items or on_empty != "send")
        # Это значит: выходим когда groups пустые И (deferred_items НЕ пустые
        #   ИЛИ on_empty НЕ "send").
        # Переписанное (эквивалентное):
        #   - выходим: on_empty="skip" + groups пустые (неважно, есть ли deferred)
        #   - выходим: on_empty="send" + groups пустые + deferred_items НЕ пустые
        #     (items oversized → run_digest::send_text→False → sent_urls=[] → OK)
        #   - НЕ выходим: on_empty="send" + groups пустые + deferred_items ПУСТЫЕ
        #     → отправляем уведомление "нет новостей"
        if not groups and (on_empty != "send" or deferred_items):
            # CRITICAL-фикс (3): все items упали при рендеринге — ранний выход,
            # но _build_failed urls ДОЛЖНЫ быть помечены 'failed', иначе они
            # вечно останутся deferred (маркеры окна двигаются при return 0,
            # но state='deferred' → при следующем окне они снова выберутся).
            # Minor 2-фикс: границу окна двигает ТОЛЬКО успех. Если
            # set_state('failed') упал — НЕ двигаем digest_window_started_at
            # (иначе оставшиеся deferred окажутся позади новой границы,
            # отсекутся published-фильтром и молча потеряются) и выходим.
            if _build_failed:
                logger.warning(
                    "Дайджест: %d items выпали при рендеринге — помечаются 'failed'",
                    len(_build_failed),
                )
                try:
                    self.storage.set_state(_build_failed, "failed")
                except Exception as e:
                    logger.warning(
                        "Дайджест: set_state('failed') упал: %s — "
                        "граница окна НЕ сдвинута, items остаются deferred",
                        e,
                    )
                    return 0
            logger.info("Дайджест: нет сообщений для отправки.")
            # M2: пишем маркер окна и при force+empty, чтобы _maybe_run_digest
            # не спамил повторно в следующем цикле.
            self.storage.set_meta("last_digest_at", digest_started_iso)
            unprocessed = self.storage.get_oldest_unprocessed_fetched_at()
            new_boundary = digest_started_iso
            if isinstance(unprocessed, str):
                udt = _iso_dt(unprocessed)
                if udt is not None and udt < now_utc:
                    new_boundary = unprocessed
            self.storage.set_meta("digest_window_started_at", new_boundary)
            return 0

        # ── M3: финальная сборка item_buttons_per_msg с изоляцией битых items ──
        # Карта url → target_id для ВСЕХ успешно созданных targets: после
        # пересборки сообщений (m1-фикс ниже) кнопки восстанавливаются по ней.
        target_id_by_url: Dict[str, Optional[int]] = {}
        if should_feedback:
            item_buttons_per_msg = []
            for msg, group_items in groups:
                item_buttons: List[dict] = []
                for item in group_items:
                    try:
                        target_id = self.storage.get_or_create_feedback_target_from_item(
                            item, "digest"
                        )
                    except Exception as e:
                        # MAJOR 2a-фикс: per-item poison pill — изолируем битый item.
                        # Он помечается 'failed' и не вернётся в следующие окна.
                        logger.warning(
                            "digest: item '%s' исключение при feedback-target: %s — "
                            "помечается 'failed'",
                            (item.title or "")[:60], e,
                        )
                        failed_digest_urls.append(item.url)
                        target_id = None
                    if target_id is not None:
                        target_id_by_url[item.url] = target_id
                    item_buttons.append({
                        "target_id": target_id,
                    })
                item_buttons_per_msg.append(item_buttons)
        else:
            item_buttons_per_msg = None  # type: ignore[assignment]

        # ── MAJOR 2-фикс: убираем failed_digest_urls из groups ДО отправки ──
        # Критично: item с упавшим target НЕ должен отправляться (он уже
        # зафиксирован как 'failed' и при sent→digested перезатрётся в 'failed').
        # Также: total_to_send пересчитывается от отфильтрованных groups.
        #
        # MAJOR m1-фикс: текст сообщения ПЕРЕСОБИРАЕТСЯ из оставшихся items.
        # Раньше msg собирался в build_digest_messages (:1730) ДО выбивания
        # item'ов, поэтому выбитый item оставался виден в тексте, а кнопок
        # для него уже не было → 👍/👎 следующих новостей визуально
        # сдвигались (неверная атрибуция фидбэка), а сам item не помечался
        # digested и дублировался в следующем окне. Теперь его нет ни в
        # тексте (включая заголовок «N из M»), ни в кнопках; он помечается
        # 'failed' ниже (:1939) и повторно не отправляется.
        if failed_digest_urls:
            failed_set = set(failed_digest_urls)
            dropped_from_text = any(
                it.url in failed_set for _, grp in groups for it in grp
            )
            if dropped_from_text:
                surviving_items = [
                    it for _, grp in groups for it in grp
                    if it.url not in failed_set
                ]
                if surviving_items:
                    groups, _, _rebuild_failed = build_digest_messages(
                        surviving_items,
                        max_items_per_message=max_items_per_message,
                    )
                    # Пересборка идемпотентна (items уже рендерились), но
                    # подстраховываемся: выпавшие и здесь — тоже 'failed'.
                    for url in _rebuild_failed:
                        if url not in failed_set:
                            failed_set.add(url)
                            failed_digest_urls.append(url)
                else:
                    groups = []
            if item_buttons_per_msg is not None:
                # Кнопки — строго по items финальных groups (по url), чтобы
                # 👍/👎 каждой новости соответствовали её target_id.
                item_buttons_per_msg = [
                    [
                        {"target_id": target_id_by_url.get(it.url)}
                        for it in grp
                    ]
                    for _, grp in groups
                ]

        # total_to_send — от отфильтрованных groups (items без упавших target).
        # При частичном сбое отправки (sent_count < total_to_send) лидеры,
        # дошедшие до получателя, помечены digest; дубли НЕ трогаем.
        total_to_send = sum(len(grp) for _, grp in groups)

        # ── M3: счётчик подряд идущих неудачных запусков дайджеста ──────────
        failure_count = 0
        try:
            raw_fail = self.storage.get_meta("digest_failure_count")
            if raw_fail is not None:
                failure_count = int(raw_fail)
        except (TypeError, ValueError):
            failure_count = 0

        if failure_count >= 3:
            # C1-фикс: 3-strike НЕ сдвигает границу окна и не теряет items.
            # Границу двигает ТОЛЬКО успешное закрытие окна (см. ниже).
            # Иначе неотправленный deferred-бэклог оставался бы позади новой
            # границы, отсекался published-фильтром и молча вымывался
            # purge_items (deferred_retention_days). Items остаются deferred
            # и повторятся; цикл ограничен digest_failure_count + retention.
            sample_urls = [it.url for _, grp in groups for it in grp][:5]
            logger.warning(
                "Дайджест: пропущен (%d неудачных попыток подряд) — "
                "%d items остаются deferred, граница окна НЕ сдвинута, "
                "счётчик сброшен; примеры url: %s",
                failure_count, total_to_send, ", ".join(sample_urls) or "—",
            )
            self.storage.set_meta("digest_failure_count", "0")
            self.storage.set_meta("last_digest_at", digest_started_iso)
            # P2-фикс: 3-strike ветка возвращалась ДО терминализации
            # poison-pill items (битый feedback-target / падение рендеринга) —
            # они вечно циклятся в 'deferred', каждый раз исключаясь из groups.
            # Помечаем их 'failed' и здесь.
            if failed_digest_urls:
                try:
                    marked = self.storage.set_state(failed_digest_urls, "failed")
                    logger.info(
                        "digest (3-strike): %d/%d items → 'failed' "
                        "(per-item исключение: рендеринг +/ feedback-target)",
                        marked, len(failed_digest_urls),
                    )
                except Exception as e:
                    logger.warning(
                        "digest (3-strike): set_state('failed') упал: %s", e,
                    )
            return 0

        # ── Отправляем через digest (с готовыми groups + buttons) ──
        try:
            sent_count, sent_urls, digest_failed_urls = _run_digest(
                storage=self.storage,
                delivery=self.delivery,
                groups=groups,
                item_buttons_per_msg=item_buttons_per_msg,
                failed_urls_in=failed_digest_urls,
            )
        except Exception as e:
            # C1-фикс: неуспешная попытка НЕ двигает digest_window_started_at —
            # границу окна сдвигает ТОЛЬКО успешное закрытие окна (ниже).
            # Раньше здесь писался digest_started_iso («сейчас»), после чего
            # published-фильтр (грейс = poll_interval) отсекал все
            # неотправленные deferred старше ~15 мин → накопленный бэклог
            # навсегда выпадал из дайджеста и тихо вымывался purge_items.
            # Зацикливание ограничено digest_failure_count (3-strike warning)
            # и штатным deferred_retention_days в purge_items.
            logger.warning(
                "Дайджест: исключение при отправке: %s — items остаются "
                "deferred, граница окна НЕ сдвинута",
                e,
            )
            # P2-фикс: shutdown — штатная остановка, не сбой. Не инкрементируем
            # счётчик и не пишем last_digest_at (см. ниже аналогичную ветку).
            if self._shutdown_event is not None and self._shutdown_event.is_set():
                logger.info(
                    "Дайджест: остановка (shutdown) — счётчик неудач и "
                    "last_digest_at НЕ изменяются",
                )
                return 0
            failure_count += 1
            self.storage.set_meta("digest_failure_count", str(failure_count))
            logger.info(
                "Дайджест: счётчик неудачных попыток %d/3",
                failure_count,
            )
            self.storage.set_meta("last_digest_at", digest_started_iso)
            return 0

        # MAJOR 2-фикс: per-item poison pill — изолируем битые items.
        # digest_failed_urls содержит: build_failed_urls (рендеринг) +
        # failed_digest_urls (feedback-target). Все они исключены из groups
        # ДО отправки (не отправлялись, не помечены digested).
        # Pipeline помечает их 'failed' — они НЕ вернутся в следующие окна.
        if digest_failed_urls:
            try:
                marked = self.storage.set_state(digest_failed_urls, "failed")
                logger.info(
                    "digest: %d/%d items → 'failed' (per-item исключение: "
                    "рендеринг +/ feedback-target)",
                    marked, len(digest_failed_urls),
                )
            except Exception as e:
                logger.warning("digest: set_state('failed') упал: %s", e)

        # Critical-фикс: retry_count для deferred НЕ инкрементируется,
        # deferred НЕ терминализируются. При транзиентном сбое они остаются
        # в очереди и повторяются в следующем окне.
        # C1-фикс: digest_window_started_at двигает ТОЛЬКО успешное закрытие
        # окна (ветка sent_count >= total_to_send ниже). Неуспешные ветки
        # пишут только last_digest_at / digest_failure_count — граница окна
        # остаётся прежней, поэтому неотправленный бэклог не выпадает из
        # published-фильтра и не теряется.
        # Retention-purge для deferred реализован в purge_items
        # (deferred_retention_days), но терминализация по retry НЕ применяется.

        # M2-фикс: last_digest_at пишется ПРИ ЛЮБОМ корректном завершении окна.
        # (a) sent>=total_to_send: все items отправлены → digested, дубли терминализированы.
        # (б) пустой deferred + on_empty=skip: окно корректно проверено,
        #     слать нечего → пишем маркер (не спамим в следующем цикле).
        # (в) пустой deferred + on_empty=send: пустое уведомление отправлено → окно обработано.
        # (г) on_empty=send + force + пустой deferred: тоже пишем маркер.
        # При транзиентном сбое last_digest_at пишется при 3-strike — чтобы не
        # спамить попытками до следующего scheduled, — но граница окна
        # НЕ сдвигается, поэтому items доживают до следующей попытки.
        now_iso = datetime.now(tz.utc).replace(microsecond=0).isoformat()
        # (a) sent_count >= total_to_send: все items дайджеста отправлены →
        #     digest.run_digest уже пометил их 'digested' через sent_urls.
        #     Терминализируем дубли (они не входят в groups) и пишем маркер окна.
        # (б) частичный сбой (sent_count < total_to_send): лидеры, дошедшие до
        #     получателя, помечены digest.run_digest; дубли НЕ трогаем — лидер
        #     остаётся deferred и повторится в следующем окне (at-least-once).
        if sent_count >= total_to_send and sent_count > 0:
            logger.info("Дайджест: %d новостей → 'digested'", sent_count)
            # M3: сброс счётчика неудачных попыток при успехе
            self.storage.set_meta("digest_failure_count", "0")
            self.storage.set_meta("last_digest_at", now_iso)
            if _dup_urls:
                try:
                    marked_dup = self.storage.set_digested(list(_dup_urls))
                    if marked_dup < len(_dup_urls):
                        logger.warning(
                            "Дайджест: дубли помечены частично (%d из %d)",
                            marked_dup, len(_dup_urls),
                        )
                except Exception as e:
                    # best-effort: не роняем дайджест. Дубли останутся deferred
                    # и будут отфильтрованы кластеризацией в следующем окне.
                    logger.warning(
                        "Дайджест: не удалось пометить дубли digested: %s", e,
                    )
            # ── Оконная очистка ──────────────────────────────────────────
            # Граница следующего окна = момент старта этого дайджеста.
            # Если остались необработанные new/pending, отодвигаем границу
            # назад к их fetched_at, чтобы они попали в следующее окно.
            unprocessed = self.storage.get_oldest_unprocessed_fetched_at()
            new_boundary = digest_started_iso
            if isinstance(unprocessed, str):
                udt = _iso_dt(unprocessed)
                if udt is not None and udt < now_utc:
                    new_boundary = unprocessed
                    logger.info(
                        "Дайджест: есть необработанные items с %s — "
                        "граница окна отодвинута", unprocessed,
                    )
            try:
                # Удаляем ровно окно: всё, что собрано ДО window_start_iso.
                # new/pending защищены на уровне storage.
                # Пришедшее во время отправки (fetched_at >= window_start_iso)
                # остаётся — уйдёт в следующее окно.
                # FIX: digest_started_iso (NOW) заменён на window_start_iso (логическая
                # граница). При bootstrap NOW-24h мог быть МЕНЬШЕ fetched_at
                # только-что сохранённого item, и он бы удалялся на первом же
                # digest. На повторных — NOW меняется, window_start_iso фиксирована
                # из meta, и item между ними тоже бы удалился. window_start_iso
                # стабильна: bootstrap фиксирует её в meta после первого digest.
                deleted = self.storage.delete_items_before(window_start_iso)
                logger.info("Дайджест: окно закрыто, удалено %d items", deleted)
            except Exception as e:
                logger.error("Дайджест: не удалось очистить окно: %s", e)
            else:
                self.storage.set_meta("digest_window_started_at", new_boundary)
        elif not deferred_items and on_empty == "send":
            # on_empty=send: отправляем пустое уведомление
            empty_msg = "📰 Дайджест — нет новых новостей."
            try:
                self.delivery.send_text(empty_msg)
                logger.info("Дайджест: пустой дайджест отправлен (on_empty=send)")
                self.storage.set_meta("last_digest_at", now_iso)
                # Закрываем окно с откатом границы при наличии необработанных.
                unprocessed = self.storage.get_oldest_unprocessed_fetched_at()
                new_boundary = digest_started_iso
                if isinstance(unprocessed, str):
                    udt = _iso_dt(unprocessed)
                    if udt is not None and udt < now_utc:
                        new_boundary = unprocessed
                self.storage.set_meta("digest_window_started_at", new_boundary)
            except Exception as e:
                logger.warning("Дайджест: пустой дайджест не отправлен: %s", e)
        else:
            # sent_count < total_to_send при непустом deferred, ИЛИ
            # total_to_send == 0 (все items выбиты фильтрацией target'ов).
            # C1-фикс: граница окна НЕ сдвигается — её двигает только успешное
            # закрытие окна. Иначе неотправленный бэклог с published старше
            # новой границы отсекался published-фильтром и молча вымывался
            # purge_items по deferred_retention_days.
            # MAJOR 2b-фикс: reset счётчика — только после 3-strike (чтобы
            # бот сам восстанавливался после разового внешнего сбоя).
            # Minor 1-фикс: счётчик 3-strike инкрементируется ТОЛЬКО при
            # полном провале доставки (sent_count == 0 AND total_to_send > 0).
            # Частичный успех (часть сообщений доставлена) — не failure:
            # иначе 3 частичных успеха подряд приводили к пропуску следующего
            # scheduled-окна. total_to_send == 0 (все items выбиты фильтрацией
            # target'ов) — тоже не failure: они помечены 'failed', а не
            # потеряны.
            if sent_count == 0 and total_to_send > 0:
                # P2-фикс: shutdown во время отправки — штатная остановка,
                # а не сбой доставки. TelegramDelivery.send_digest при
                # _shutdown_event возвращает False (внутри _api_call —
                # TelegramDeliveryError("shutdown…")), и раньше это считалось
                # провалом: 3 деплоя подряд → 3-strike → пропуск окна.
                # Не инкрементируем digest_failure_count и не пишем
                # last_digest_at — items остаются deferred до следующего окна.
                if self._shutdown_event is not None and self._shutdown_event.is_set():
                    logger.info(
                        "Дайджест: остановка (shutdown) во время отправки — "
                        "%d items остаются deferred, digest_failure_count и "
                        "last_digest_at НЕ изменяются",
                        total_to_send,
                    )
                else:
                    failure_count += 1
                    self.storage.set_meta("digest_failure_count", str(failure_count))
                    if failure_count >= 3:
                        remaining = max(0, total_to_send - sent_count)
                        sample_urls = [
                            it.url for _, grp in groups for it in grp
                            if it.url not in set(sent_urls)
                        ][:5]
                        logger.warning(
                            "Дайджест: %d неудачных попыток подряд — %d items остаются "
                            "deferred, граница окна НЕ сдвинута (повтор в следующем окне), "
                            "счётчик сброшен; примеры url: %s",
                            failure_count, remaining, ", ".join(sample_urls) or "—",
                        )
                        # reset после 3-strike — бот сам восстанавливается после
                        # разового внешнего сбоя (сеть/Telegram API transient).
                        self.storage.set_meta("digest_failure_count", "0")
                        self.storage.set_meta("last_digest_at", now_iso)
                    else:
                        logger.debug(
                            "Дайджест: пропущен (sent_count=%d/%d, failures=%d/3)",
                            sent_count, total_to_send, failure_count,
                        )
            else:
                logger.debug(
                    "Дайджест: частичная доставка или пустой набор "
                    "(sent_count=%d/%d) — 3-strike не инкрементируется",
                    sent_count, total_to_send,
                )

        return sent_count

    def _maybe_run_digest(self) -> None:
        """
        Проверяет расписание дайджеста (digest.times в локальной tz).

        T4-фикс (docstring ↔ код): актуальная логика:

          1. Для каждого времени t из times:
               scheduled сегодня или вчера (ближайший <= now).
          2. Окно «необработано» если:
               — last_digest_at < scheduled  (по timestamp)
          3. Если now >= scheduled И окно не обработано → запускаем.
             НА СВЕЖЕЙ БД (last_digest_at пуст): дайджест отправляется
             только если текущее время >= scheduled. Раньше scheduled
             — не отправляется (catch-up до scheduled не выполняется).

          Это гарантирует: нет пропусков (цикл в любую минуту после
          scheduled отработает) и нет дублей (last_digest_at блокирует
          повтор).
        """
        digest_cfg = self.config.get("digest", {}) or {}
        if not _parse_bool(digest_cfg.get("enabled", True)):
            return

        times = digest_cfg.get("times", ["09:00", "21:00"])
        if not times:
            return

        now_local = datetime.now(tz.utc).astimezone()
        last_raw = self.storage.get_meta("last_digest_at")
        last_digest: datetime | None = None
        if last_raw:
            try:
                last_digest = datetime.fromisoformat(last_raw).astimezone()
            except Exception:
                pass

        for t in times:
            try:
                sched_hour, sched_min = int(t[:2]), int(t[3:])
            except (ValueError, IndexError):
                logger.warning("Дайджест: невалидное время %r в times", t)
                continue

            # Сегодняшний scheduled момент (local tz)
            sched_today = now_local.replace(
                hour=sched_hour, minute=sched_min, second=0, microsecond=0
            )

            # ── Выбор окна для отправки ────────────────────────────
            # (1) Свежая БД (last_digest is None): только сегодняшние уже
            #     прошедшие окна. Вчерашние НЕ рассматриваются — не шлём
            #     дайджест «на всякий случай» до первого scheduled.
            # (2) Непустой last_digest: catch-up пропущенных окон — ищем
            #     ближайший прошедший момент (today или yesterday), который
            #     старше last_digest. Это покрывает пропущенные циклы
            #     и задержанные запуски (now > scheduled).
            if last_digest is None:
                # Свежая БД: только сегодняшние уже прошедшие окна
                if sched_today > now_local:
                    continue  # будущее — пропускаем
                actual_scheduled = sched_today
            else:
                # Catch-up: вчерашнее и/или сегодняшнее, whichever <= now_local
                sched_yesterday = sched_today - timedelta(days=1)
                candidates = [c for c in [sched_today, sched_yesterday] if c <= now_local]
                if not candidates:
                    continue
                actual_scheduled = max(candidates)

            sched_key = actual_scheduled.strftime("%H:%M")

            # Окно обработано если last_digest_at >= actual_scheduled.
            # Сравниваем через timestamp() — корректно работает с разными tzinfo.
            if last_digest is not None and \
                    last_digest.timestamp() >= actual_scheduled.timestamp():
                logger.debug(
                    "Дайджест: scheduled=%s уже обработан (last=%s)",
                    sched_key, last_raw,
                )
                continue

            # Окно не обработано: запускаем.
            # N2: запуск возможен даже если цикл попал в окно с задержкой
            # (now > scheduled). last_digest_at блокирует повторный запуск.
            logger.info(
                "Дайджест: scheduled=%s наступил (now=%s), выполняем...",
                sched_key, now_local.strftime("%H:%M"),
            )
            try:
                self.run_digest(force=False)
            except Exception as e:
                logger.error("Дайджест: исключение: %s", e, exc_info=True)
            break  # Одно окно за цикл — достаточно

    def _run_delivery_side_effects(self) -> None:
        """
        Major #1: единая точка проверки паузы для дайджеста и баланс-алерта.
        Вызывается ПОСЛЕ основных батчей (и в normal path, и в off-peak path).
        При паузе пропускает оба вызова, логирует; purge по-прежнему выполняется
        в вызывающем коде. Каждый эффект обёрнут в отдельный try/except:
        падение одного не блокирует другой.
        """
        if self._is_paused():
            logger.info(
                "PAUSED: дайджест и баланс-алерт пропущены (выполняется очистка)"
            )
            return
        try:
            self._maybe_check_balance()
        except Exception as e:
            logger.warning("_maybe_check_balance ошибка: %s", e)
        try:
            self._maybe_run_digest()
        except Exception as e:
            logger.warning("_maybe_run_digest ошибка: %s", e)

    def _maybe_check_balance(self) -> None:
        """
        Проверяет баланс LLM-аккаунта не чаще check_interval_hours.
        При низком балансе отправляет алерт владельцу.
        """
        balance_cfg = self.config.get("llm", {}).get("balance") or {}
        if not _parse_bool(balance_cfg.get("enabled", True)):
            return

        if self.llm_filter is None:
            return

        check_interval_hours = float(balance_cfg.get("check_interval_hours", 24))
        threshold = float(balance_cfg.get("threshold", 2.0))

        # гейтинг по мета-ключу llm_balance_last_check
        last_check_raw = self.storage.get_meta("llm_balance_last_check")
        if last_check_raw is not None:
            try:
                last_check = datetime.fromisoformat(last_check_raw)
                elapsed_hours = (
                    datetime.now(tz.utc) - last_check
                ).total_seconds() / 3600
                if elapsed_hours < check_interval_hours:
                    return
            except Exception:
                pass  # невалидное значение — считаем что не проверяли

        try:
            balance: BalanceInfo = self.llm_filter.get_balance()
        except Exception as e:
            logger.warning("Ошибка get_balance(): %s", e)
            return

        # сохраняем время проверки
        self.storage.set_meta(
            "llm_balance_last_check",
            datetime.now(tz.utc).isoformat(),
        )

        if balance.is_available and balance.total_balance < threshold:
            # cooldown: не чаще check_interval_hours
            last_alert_raw = self.storage.get_meta("llm_balance_last_alert")
            should_alert = True
            if last_alert_raw is not None:
                try:
                    last_alert = datetime.fromisoformat(last_alert_raw)
                    elapsed_hours = (
                        datetime.now(tz.utc) - last_alert
                    ).total_seconds() / 3600
                    if elapsed_hours < check_interval_hours:
                        should_alert = False
                except Exception:
                    pass  # невалидное значение — шлём алерт

            if should_alert:
                currency = balance.currency or "USD"
                alert_text = (
                    f"⚠️ Низкий баланс DeepSeek: {balance.total_balance} {currency} "
                    f"(порог {threshold}). Пополни: https://platform.deepseek.com"
                )
                if self.delivery.send_text(alert_text):
                    self.storage.set_meta(
                        "llm_balance_last_alert",
                        datetime.now(tz.utc).isoformat(),
                    )
                    logger.info("Алерт о низком балансе DeepSeek отправлен")

    def close(self) -> None:
        self.storage.close()
        self.delivery.close()
        # закрываем LLM-клиент (идемпотентно)
        if self.llm_filter is not None:
            try:
                self.llm_filter.client.close()
            except Exception:
                pass