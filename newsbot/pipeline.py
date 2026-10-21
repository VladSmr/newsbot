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


def _parse_tags(tags_str: Optional[str]) -> List[str]:
    """Парсит JSON-массив тегов из БД."""
    if not tags_str:
        return []
    try:
        import json
        return json.loads(tags_str)
    except Exception:
        return []

def _infer_category(item) -> str:
    """
    Определяет категорию новости по source + tags.
    categories: cinema, games, space, tech, cyber, science
    """
    source = (item.source or "").lower()
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
    if source in ('tom shardware', 'habr', 'techcrunch'):
        return 'tech'
    if any(t in tags_lower for t in ('technology', 'ai', 'tech', 'software', 'hardware')):
        return 'tech'

    # Кибербезопасность
    if source in ('krebs on security', 'bleepingcomputer'):
        return 'cyber'
    if any(t in tags_lower for t in ('cyber', 'security', 'malware', 'hack', 'vulnerability')):
        return 'cyber'

    # Наука
    if source in ('science daily', 'phys.org', 'n+1', 'naked science', 'statnews'):
        return 'science'
    if any(t in tags_lower for t in ('science', 'research', 'study', 'discovery', 'medical', 'health')):
        return 'science'

    return 'other'


def _cap_by_location(
    items: List[NewsItem],
    max_per_location: int = 2,
) -> List[NewsItem]:
    """
    Ограничивает число новостей по локации и дню.

    Для каждой пары (локация, день публикации) оставляет не более
    max_per_location новостей. Items с cluster_key=None или корзиной
    'other' пропускаются (кэп на них не распространяется).

    Алгоритм: итерация по порядку, для каждой (локация, день) сохраняются
    первые max_per_location items — остальные отфильтровываются.
    Вызывается в run_digest ПОСЛЕ backfill, ДО _category_guarantee.
    """
    if max_per_location <= 0 or not items:
        return list(items)

    location_day_counts: Dict[tuple, int] = {}
    result: List[NewsItem] = []

    for item in items:
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
        if count < max_per_location:
            result.append(item)
            location_day_counts[key] = count + 1

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

    Сорт вывода — по llm_score DESC вызывающей стороной после вызова.
    """
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
    def __init__(self, config: dict):
        self.config = config
        self.sources = config.get("sources", []) or []

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
            return TelegramDelivery(token=token, chat_id=chat_id)
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
            flt = LLMFilter(llm_cfg)
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
                    self._apply_result(item, res, push_th, digest_th,
                                       push_urgent_only, skipped_urls, deferred_urls,
                                       to_deliver, llm_batch_records)

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

                    for item, res in zip(fresh_batch, batch_results):
                        # сохраняем результат в item (всегда, даже ok=False)
                        item.llm_score = res.score
                        item.llm_summary = res.summary
                        item.llm_reason = res.reason
                        self._apply_result(item, res, push_th, digest_th,
                                           push_urgent_only, skipped_urls, deferred_urls,
                                           to_deliver, llm_batch_records)

            logger.debug("LLM-вызовов за батч: %d", llm_call_count)

            # сохраняем все llm-результаты батчем (push + deferred + skipped)
            if llm_batch_records:
                self.storage.update_llm_results_batch(llm_batch_records)

            # ── Pre-LLM кластеризация: распределить максимальный score по кластеру ──
            from collections import defaultdict
            cluster_scores: Dict[str, float] = defaultdict(lambda: 0.0)
            cluster_members: Dict[str, List[NewsItem]] = defaultdict(list)

            for item in items:
                if getattr(item, "cluster_key", None):
                    cluster_members[item.cluster_key].append(item)
                    if item.llm_score is not None and item.llm_score > cluster_scores[item.cluster_key]:
                        cluster_scores[item.cluster_key] = float(item.llm_score)

            # Распределить максимальный score обратно по кластеру
            for key, members in cluster_members.items():
                best_score = cluster_scores[key]
                for item in members:
                    if item.llm_score is None or item.llm_score < best_score:
                        item.llm_score = int(best_score)

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
                        item.title[:60], e,
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
        deferred_items = self.storage.get_deferred_items(
            # M3-фикс: берём с запасом, чтобы Python-фильтр по published
            # не «съел» нужные items (SQL LIMIT отрабатывает ДО фильтра).
            # Окончательная обрезка — deferred_items[:max_items] ниже.
            limit=max_items * 2 if (max_items and max_items > 0) else None,
            since_iso=window_start_iso if raw_boundary is not None else None,
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

        if raw_boundary is not None:
            ws_dt = _iso_dt(window_start_iso)
            if ws_dt is not None:
                # M2-фикс: правильная нормализация aware→naive через astimezone,
                # а не .replace(tzinfo=None) (последнее давало ошибку ±offset часов).
                # Minor 1: если ws_dt naive (legacy-meta без offset), astimezone
                # трактовала бы его как локальное время → граница съезжает на offset.
                # Лечим: naive оставляем как есть, aware нормализуем.
                ws_naive = ws_dt if ws_dt.tzinfo is None else ws_dt.astimezone(tz.utc).replace(tzinfo=None)
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
                        if pub_dt_norm >= ws_naive:
                            filtered.append(it)
                deferrable_post_published = filtered

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
        category_min = digest_cfg.get("category_min", {})
        if category_min:
            deferred_items = _category_guarantee(deferred_items, category_min, pool=category_pool)

        # MAJOR: cap by location/day — СТРОГО ПОСЛЕДНИЙ, после всех свапов.
        # Инвариант «≤max_per_location на (локация, день)» держится железно,
        # cap не может обнулить список (max_per_location>0 → в каждой группе ≥1).
        max_per_location = digest_cfg.get("max_per_location", 2)
        if max_per_location and max_per_location > 0 and deferred_items:
            deferred_items = _cap_by_location(
                deferred_items, max_per_location=max_per_location,
            )
            if deferred_items:
                logger.info(
                    "Дайджест: location-cap оставил %d items",
                    len(deferred_items),
                )

        # Финальный сорт — по llm_score DESC (состав уже зафиксирован гарантийными
        # свапами; сорт только порядок вывода). Финальный slice ПОСЛЕ гарантии.
        deferred_items.sort(key=lambda x: x.llm_score or 0, reverse=True)
        deferred_items = deferred_items[:max_items]

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

        groups, _ = build_digest_messages(
            deferred_items,
            max_items_per_message=max_items_per_message,
        )

        # Major 2: считаем сколько items реально уйдёт (для проверки полного успеха).
        # total_to_send — items в groups (отфильтрованные дубли не попадают сюда).
        # Критерий полного успеха дайджеста: все items из groups отправлены
        # (sent_count >= total_to_send). При частичном сбое
        # (sent_count < total_to_send) — лидер(ы) помечены digest.run_digest,
        # дубли НЕ терминализируются (retry в след.окне).
        total_to_send = sum(len(grp) for _, grp in groups)

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

        item_buttons_per_msg: List[List[dict]]
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
                        logger.warning(
                            "feedback: не удалось создать target для '%s': %s",
                            item.title[:60], e,
                        )
                        target_id = None
                    # NOTE: поле "url" намеренно НЕ передаётся — send_digest()
                    # использует только target_id. Ссылка на новость уже в тексте
                    # дайджеста (<a href> в digest.py footer); дублирование
                    # кнопкой отключено (URL-кнопка сломала бы экранирование).
                    item_buttons.append({
                        "target_id": target_id,
                    })
                item_buttons_per_msg.append(item_buttons)
        else:
            # ConsoleDelivery или feedback выключен — пустые кнопки
            item_buttons_per_msg = None  # type: ignore[assignment]

        # ── Отправляем через digest (с готовыми groups + buttons) ──
        sent_count, sent_urls = _run_digest(
            storage=self.storage,
            delivery=self.delivery,
            groups=groups,
            item_buttons_per_msg=item_buttons_per_msg,
        )

        # Critical-фикс: retry_count для deferred НЕ инкрементируется,
        # deferred НЕ терминализируются. При транзиентном сбое они остаются
        # в очереди и повторяются в следующем окне. last_digest_at пишется
        # только при успешной отправке (sent_count>0), обеспечивая retry.
        # Retention-purge для deferred реализован в purge_items
        # (deferred_retention_days), но терминализация по retry НЕ применяется.

        # M2-фикс: last_digest_at пишется ПРИ ЛЮБОМ корректном завершении окна.
        # (a) sent>=total_to_send: все items отправлены → digested, дубли терминализированы.
        # (б) пустой deferred + on_empty=skip: окно корректно проверено,
        #     слать нечего → пишем маркер (не спамим в следующем цикле).
        # (в) пустой deferred + on_empty=send: пустое уведомление отправлено → окно обработано.
        # (г) on_empty=send + force + пустой deferred: тоже пишем маркер.
        # При транзиентном сбое (все отправки failed) окно НЕ помечается
        # обработанным — retry в следующем цикле, а не через 12ч.
        now_iso = datetime.now(tz.utc).replace(microsecond=0).isoformat()
        # (a) sent_count >= total_to_send: все items дайджеста отправлены →
        #     digest.run_digest уже пометил их 'digested' через sent_urls.
        #     Терминализируем дубли (они не входят в groups) и пишем маркер окна.
        # (б) частичный сбой (sent_count < total_to_send): лидеры, дошедшие до
        #     получателя, помечены digest.run_digest; дубли НЕ трогаем — лидер
        #     остаётся deferred и повторится в следующем окне (at-least-once).
        if sent_count >= total_to_send and sent_count > 0:
            logger.info("Дайджест: %d новостей → 'digested'", sent_count)
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
                # Удаляем ровно окно: всё, что собрано ДО старта дайджеста
                # (отправленное, отсеянное фильтром, не влезшее в лимит).
                # new/ pending защищены на уровне storage.
                # Пришедшее во время отправки (fetched_at >= digest_started_iso)
                # остаётся — уйдёт в следующее окно.
                deleted = self.storage.delete_items_before(digest_started_iso)
                logger.info("Дайджест: окно закрыто, удалено %d items", deleted)
            except Exception as e:
                logger.error("Дайджест: не удалось очистить окно: %s", e)
            else:
                self.storage.set_meta("digest_window_started_at", new_boundary)
        elif not deferred_items and on_empty == "skip":
            # M2: окно проверено, слать нечего → пишем маркер чтобы
            # _maybe_run_digest не запускал повторно в следующем цикле.
            logger.debug("Дайджест: нет отложенных новостей, окно проверено (skip)")
            self.storage.set_meta("last_digest_at", now_iso)
            # Закрываем окно с откатом границы при наличии необработанных.
            unprocessed = self.storage.get_oldest_unprocessed_fetched_at()
            new_boundary = digest_started_iso
            if isinstance(unprocessed, str):
                udt = _iso_dt(unprocessed)
                if udt is not None and udt < now_utc:
                    new_boundary = unprocessed
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
            # sent_count==0 при непустом deferred — транзиентный сбой,
            # маркер НЕ пишем, retry в следующем цикле.
            logger.debug("Дайджест: пропущен (empty, force=%s)", force)

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