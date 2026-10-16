r"""
Дедупликация: обёртка над Storage.

Дедуп по двум уровням (оба конфигурируемые):
  Уровень 1 — глобальный точный дедуп по title_hash (без source).
    Активируется: dedup.cross_source=true (дефолт True).
    Окно: dedup.window_days (дефолт 3 дня).
  Уровень 2 — нечёткое сравнение (SequenceMatcher) кросс-источник.
    Активируется: dedup.fuzzy=true (дефолт True).
    Порог: dedup.threshold (дефолт 0.95).
    Доп. защита от ложных склеек:
      • числовой guard: мультимножество цифровых последовательностей
        нормализованного текста (жадный \d+). Жёсткий запрет при расхождении.
        NOTE: десятичные точки, запятые, знаки и % схлопываются при нормализации
        (title_normalized удаляет [^\w\s]), поэтому в текст попадают только
        целые последовательности цифр.
      • token-guard: Jaccard значащих токенов ≥ dedup.token_threshold (дефолт 0.8).

  Итоговая формула «дубль»:
    дубль ⟺ (числа совпадают) И (Jaccard ≥ token_threshold) И (ratio ≥ threshold)

  Guard'ы вычисляются до SequenceMatcher.ratio() — дешёвые проверки раньше
  дорогой. Это позволяет отсеять заведомо разные пары без тяжёлого ratio.

  Семантика: fuzzy-дедуп выполняется только при cross_source=true (кросс-источниковый
  контекст). При cross_source=false он выключается, даже если fuzzy=true явно задан.

NOTE: дедуп по точному URL (primary key) — permanent, всегда активен.
"""
import logging
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import List, Optional, Set, Tuple

from .collector import NewsItem
from .storage import Storage, MIN_TITLE_LEN

logger = logging.getLogger("newsbot.dedup")

# ─── Предфильтр для fuzzy-дедупа ─────────────────────────────────────────────
# Максимум кандидатов для сравнения SequenceMatcher (ограничение O(N)).
MAX_FUZZY_CANDIDATES = 200

# Минимальная абсолютная разница в количестве слов: если |len(a) - len(b)| > MAX_LEN_DIFF,
# ratio гарантированно < threshold даже при полном совпадении токенов.
MAX_LEN_DIFF = 15

# Минимальное пересечение токен-сетов: ratio = |tokens_a ∩ tokens_b| / min(|a|, |b|)
MIN_TOKEN_INTERSECTION = 0.3

# Регулярка для числовых токенов (числа с опциональной дробной частью).
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

# Стоп-слова (короткие и неинформативные), исключаются из token-guard.
_STOP_WORDS = {"the", "and", "for", "that", "this", "with", "from", "were", "was",
               "have", "has", "had", "their", "they", "been", "being", "also",
               "are", "but", "not", "which", "some", "into", "only", "than",
               "more", "most", "after", "over", "such", "then", "about"}


def _number_multiset(text: str) -> Counter:
    """
    Мультимножество цифровых последовательностей из нормализованного текста.
    Жёсткий guard: разные мультимножества → НЕ дубль.
    "\d+(?:[.,]\\d+)?" на нормализованном тексте = просто \d+ (точки/запятые
    уже удалены normalize), но регекс сохранён для совместимости.
    Counter: "70 70" и "70" — РАЗНЫЕ мультимножества (2 vs 1).
    """
    return Counter(_NUMBER_RE.findall(text))


def _meaningful_tokens(text: str) -> Set[str]:
    """
    Значимые токены: слова длиной ≥4, без чисел и стоп-слов.
    Используются для token-guard (Jaccard).
    """
    words = text.split()
    return {
        w for w in words
        if len(w) >= 4
        and not w.isdigit()
        and w not in _STOP_WORDS
    }


def _token_jaccard(a: str, b: str, token_threshold: float) -> bool:
    """
    Token-guard: Jaccard значащих токенов.
    Jaccard = |A ∩ B| / |A ∪ B|.
    Если Jaccard < token_threshold → разные сущности → НЕ дубль.
    """
    set_a = _meaningful_tokens(a)
    set_b = _meaningful_tokens(b)
    if not set_a or not set_b:
        # Если нет значащих токенов — пропускаем (guard не блокирует)
        return True
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return (intersection / union) >= token_threshold


def _token_set(text: str) -> Set[str]:
    """Токен-сет нормализованного текста (набор уникальных слов)."""
    return set(text.split())


def _quick_filter(norm: str, candidates: List[Tuple]) -> List[Tuple]:
    """
    Предфильтр по длине и токен-сету.
    Возвращает подмножество candidates, прошедших предфильтр.
    O(N) без SequenceMatcher — быстрый отсев заведомо непохожих.
    """
    tokens = _token_set(norm)
    if not tokens:
        return []

    norm_words = norm.split()

    filtered: List[Tuple] = []
    for title_hash, candidate_norm, *_ in candidates:
        if not candidate_norm:
            continue
        cand_words = candidate_norm.split()

        # (1) фильтр по длине
        cand_len_range = (len(cand_words) - MAX_LEN_DIFF, len(cand_words) + MAX_LEN_DIFF)
        if not (cand_len_range[0] <= len(norm_words) <= cand_len_range[1]):
            continue

        # (2) фильтр по токен-сету
        cand_tokens = _token_set(candidate_norm)
        if not cand_tokens:
            continue
        intersection = len(tokens & cand_tokens)
        token_ratio = intersection / min(len(tokens), len(cand_tokens))
        if token_ratio < MIN_TOKEN_INTERSECTION:
            continue

        filtered.append((title_hash, candidate_norm))

    return filtered[:MAX_FUZZY_CANDIDATES]


def _is_fuzzy_similar(
    a: str, b: str, threshold: float, token_threshold: float,
) -> bool:
    """
    Сравнивает два нормализованных заголовка через SequenceMatcher
    с дополнительной защитой от ложных склеек.

    Формула «дубль» (guard'ы ДО SequenceMatcher — дешёвые раньше дорогих):
        числа совпадают (мультимножество)  — жёсткий запрет при расхождении
        И token_Jaccard ≥ token_threshold  — значимые токены похожи
        И ratio ≥ threshold               — SequenceMatcher ratio

    Все три условия должны выполняться.

    NOTE: числовой guard работает на нормализованном тексте. Поскольку
    title_normalized() удаляет [^\w\s], десятичные точки, запятые, знаки
    и % уже вырезаны. Мультимножество (Counter) отличает «70 70» от «70».

    Args:
        a: нормализованный заголовок нового item
        b: нормализованный заголовок кандидата из БД
        threshold: порог SequenceMatcher ratio (дефолт 0.95)
        token_threshold: порог Jaccard значащих токенов (дефолт 0.8)
    """
    if not a or not b:
        return False

    # ── (1) Числовой guard — мультимножество (до SequenceMatcher) ─────
    # NOTE: на нормализованном тексте [.,] уже удалены, поэтому регекс
    # \d+(?:[.,]\d+)? по сути ловит только \d+. Counter считает кратности:
    # "70 70" vs "70" — разные мультимножества, жёсткий отказ.
    nums_a = _number_multiset(a)
    nums_b = _number_multiset(b)
    if nums_a != nums_b:
        return False

    # ── (2) Token guard — Jaccard значащих токенов ────────────────────
    if not _token_jaccard(a, b, token_threshold):
        return False

    # ── (3) SequenceMatcher ratio (дорогой — последним) ───────────────
    ratio = SequenceMatcher(None, a, b).ratio()
    if ratio < threshold:
        return False

    return True


def _is_meaningful_title(item: NewsItem) -> bool:
    """Заголовок достаточно осмысленный для title-based дедупа (>= MIN_TITLE_LEN)."""
    return len(item.title_normalized()) >= MIN_TITLE_LEN


def filter_new(
    items: List[NewsItem],
    storage: Storage,
    dedup_config: Optional[dict] = None,
) -> Tuple[List[NewsItem], int, int]:
    """
    Фильтрует список items, оставляя только новые (не дубли).

    Трёхуровневая проверка:
      (a) точный URL в БД — permanent dedup (URL не меняется)
      (b) нормализованный заголовок + source в пределах TTL
          (per-source дедуп, существующее поведение)
      (c) [опц.] кросс-источниковый точный дедуп: title_hash БЕЗ source
          в пределах dedup.window_days (cross_source=true)
      (d) [опц.] нечёткое сравнение (SequenceMatcher) кросс-источник,
          только для осмысленных заголовков, ТОЛЬКО когда cross_source=true
          (fuzzy=true И cross_source=true)

    Формула fuzzy-дубля (guard'ы — до SequenceMatcher):
        дубль ⟺ (числа совпадают) И (Jaccard ≥ token_threshold) И (ratio ≥ threshold)

    При нахождении дубля уровня (c) или (d) — item сохраняется в items
    со state='skipped' (URL-дедуп в следующих циклах + не попадает в delivery).

    Возвращает (new_items, total_count, dup_count).
    """
    cfg = dedup_config if dedup_config is not None else {}
    cross_source = cfg.get("cross_source", True)
    fuzzy = cfg.get("fuzzy", True)
    # NOTE: fuzzy выполняется ТОЛЬКО при cross_source=true (кросс-источниковый контекст).
    # При cross_source=false fuzzy игнорируется — это сохраняет семантику per-source
    # дедупа (иначе возникает неочевидное поведение: разные source, но fuzzy всё склеивает).
    if cross_source:
        fuzzy_enabled = fuzzy
    else:
        fuzzy_enabled = False  # явно выключаем при cross_source=false

    threshold = float(cfg.get("threshold", 0.95))
    token_threshold = float(cfg.get("token_threshold", 0.8))
    window_days = int(cfg.get("window_days", 3))

    # clamp thresholds
    threshold = max(0.0, min(1.0, threshold))
    token_threshold = max(0.0, min(1.0, token_threshold))

    new_items: List[NewsItem] = []
    dup_count = 0
    cross_source_skipped = 0   # для единого лога за цикл
    fuzzy_skipped = 0          # для единого лога за цикл

    # Множество ключей (normalized_title, source), встреченных в этом цикле.
    seen_in_cycle: Set[tuple] = set()

    # M2-фикс: кэш запросов get_recent_normalized_titles по exclude_source.
    # Таблица seen_titles не меняется в этом цикле — результат стабилен.
    # Один запрос на уникальный exclude_source (а не O(N) запросов).
    _recent_cache: dict = {}

    # Предварительно собираем множество source'ов, которые встретятся
    # в цикле, чтобы запросить их ДО начала основного цикла.
    # Это позволяет сделать все запросы заранее, а не по требованию.
    if cross_source or fuzzy_enabled:
        for item in items:
            if _is_meaningful_title(item):
                src = item.source
                if src not in _recent_cache:
                    _recent_cache[src] = storage.get_recent_normalized_titles(
                        window_days=window_days,
                        limit=500,
                        exclude_source=src,
                    )

    for item in items:
        # (a) точный URL в БД
        if storage.is_duplicate(item):
            dup_count += 1
            logger.debug("URL-дубликат пропущен: %s", item.url[:80])
            continue

        normalized = item.title_normalized()

        # (b) title+source в БД — проверено внутри is_duplicate
        # (c) title+source ВНУТРИ цикла
        if _is_meaningful_title(item):
            cycle_key = (normalized, item.source)
            if cycle_key in seen_in_cycle:
                dup_count += 1
                logger.debug(
                    "Внутрицикловой дубликат пропущен: title=%r source=%s",
                    item.title[:60], item.source,
                )
                continue
            seen_in_cycle.add(cycle_key)

        # ── Кросс-источниковые кандидаты — из кэша (O(1) lookup) ──
        recent: List[tuple] = []
        if (cross_source or fuzzy_enabled) and _is_meaningful_title(item):
            recent = _recent_cache.get(item.source, [])

        # ── Уровень 1: кросс-источниковый точный дедуп ──
        # Проверяется ЯВНО в filter_new (а не внутри is_duplicate), чтобы
        # при срабатывании сохранить item как skipped.
        is_cross_source_dup = False
        if cross_source and recent:
            # предфильтр по title_hash (точный) — быстрая проверка O(N)
            for title_hash, *_ in recent:
                if title_hash == normalized:
                    is_cross_source_dup = True
                    break

        if is_cross_source_dup:
            storage.save_skipped(item)
            cross_source_skipped += 1
            dup_count += 1
            logger.debug(
                "Cross-source дубль пропущен: title=%r source=%s",
                item.title[:60], item.source,
            )
            continue

        # ── Уровень 2: fuzzy-дедуп (только при cross_source=true) ──
        fuzzy_duplicate = False
        if fuzzy_enabled and recent:
            # предфильтр: быстрый отсев по длине и токен-сету
            filtered = _quick_filter(normalized, recent)
            for _, cand_norm, *_ in filtered:
                if _is_fuzzy_similar(
                    normalized, cand_norm, threshold, token_threshold,
                ):
                    fuzzy_duplicate = True
                    break

        if fuzzy_duplicate:
            storage.save_skipped(item)
            fuzzy_skipped += 1
            dup_count += 1
            logger.debug(
                "Fuzzy-дубликат пропущен: title=%r (similar to recent title)",
                item.title[:60],
            )
            continue

        new_items.append(item)

    if cross_source_skipped:
        logger.info("Дедуп: отфильтровано %d cross-source дублей", cross_source_skipped)
    if fuzzy_skipped:
        logger.info("Дедуп: отфильтровано %d fuzzy-дублей", fuzzy_skipped)

    return new_items, len(items), dup_count