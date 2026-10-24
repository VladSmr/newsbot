r"""
Дедупликация: обёртка над Storage.

Внутри-доменный дедуп (Задача 2.4):
  Если один и тот же cluster_key (тема+день) за последние 2 часа,
  независимо от source — пропускаем новый item как потенциальный дубль.
  Выполняется ДО fuzzy-дедупа (быстрая отсечка).

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
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import List, Optional, Set, Tuple

from .collector import NewsItem
from .storage import Storage, MIN_TITLE_LEN

logger = logging.getLogger("newsbot.dedup")


# ─── Внутри-доменный дедуп (Задача 2.4) ──────────────────────────────────────




def _domain_recent_duplicate(item: NewsItem, storage: Storage, hours: int = 2) -> bool:
    """
    Быстрый дедуп по cluster_key: если ОДИН cluster_key (тема+день)
    уже был за последние hours — блокируем. Разные темы от одного
    источника НЕ блокируются.

    M2-фикс: сравниваем по суффиксу stem|day (без локационного префикса).
    При ингесте item получает stub-ключ `other|stem|day` (loc=other, stub),
    после скоринга pipeline._process_batch пересобирает ключ с LLM-локацией
    → `moscow|stem|day` (loc=moscow, пересобранный).
    При прямом сравнении `other|abc|day` != `moscow|abc|day` → guard не срабатывает.
    Инвариант guard: ловит «та же тема + тот же день», независимо от того,
    пересобран ли ключ в БД. Сравнение по суффиксу `%|stem|day` обеспечивает
    этот инвариант — prefix (location) игнорируется.

    Fallback на exact match: для старых ключей без `|` (например `cb_rate_2024`)
    суффикс неприменим, используем точное совпадение.

    NOTE: суффиксный LIKE с leading wildcard = full scan по cluster_key.
    Вызывается один раз на item в filter_new; при ~10k items/день и 2h-окне
    сканируется ~800 строк — приемлемо. Дополнительный фильтр по pub_day
    в SQL не применяется (pub_day извлекается из cluster_key через Python-
    split, а не из отдельного столбца; без отдельного индекса по pub_day
    дополнительный фильтр не даёт выигрыша, только усложняет запрос).

    NOTE: 32-битный stem_hash (8 hex). При ~10k заголовков/день вероятность
    коллизии ≈ 10^-5 (birthday paradox: ~N^2/2^32 ≈ 10^8/4*10^9 ≈ 2.5e-2,
    но stem_hash = md5[:8] = 32 бита, поэтому P ≈ N^2/2^(32+1) ≈ 10^8/8.6e9
    ≈ 1.2e-2 ≈ 1.2%). При коллизии guard может пропустить нерелевантную
    новость как дубль — осознанный компромисс (широкий stem менее агрессивен,
    чем точный match, и не приводит к потере новостей).
    """
    cluster_key = getattr(item, 'cluster_key', None)
    if not cluster_key:
        return False
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        conn = storage._connect()
        if '|' in cluster_key:
            # Новый формат: `loc|stem|day` — сравниваем по суффиксу stem|day.
            # Отрезаем локационный префикс: suffix = "stem|day".
            # Паттерн `'%' || ? ESCAPE '\'` = suffix match (LIKE с leading wildcard).
            # ESCAPE '\' защищает от метасимволов %/_ в stem (теоретически,
            # md5-hex не содержит их, но защита обязательна).
            # Пример: cluster_key='other|abc12345|2026-10-21'
            #   → suffix='abc12345|2026-10-21'
            #   → паттерн '%abc12345|2026-10-21'
            #   → MATCH: 'other|abc12345|2026-10-21', 'moscow|abc12345|2026-10-21'
            #   → NO MATCH: 'other|def99999|2026-10-21', 'moscow|def99999|2026-10-21'
            suffix = cluster_key.split('|', 1)[1]
            row = conn.execute(
                "SELECT 1 FROM items WHERE cluster_key LIKE '%' || ? ESCAPE '\\' AND fetched_at>=? LIMIT 1",
                (suffix, cutoff),
            ).fetchone()
        else:
            # Старый формат без `|`: точное совпадение.
            row = conn.execute(
                "SELECT 1 FROM items WHERE cluster_key=? AND fetched_at>=? LIMIT 1",
                (cluster_key, cutoff),
            ).fetchone()
        return row is not None
    except Exception as e:
        logger.warning("Доменный дедуп: ошибка, fail-open: %s", e)
        return False


# ─── Защищённый разбор dedup.* из конфига ─────────────────────────────────────
# Опечатка в config.yaml (window_days: 3d, threshold: high, cross_source: "false")
# не должна ронять filter_new внутри цикла (иначе каждый цикл обрывается до
# save_many → новости не сохраняются). Невалидное значение → warning + дефолт.

# Границы «разумности» для окна дедупа (дней): 0/отрицательное молча выключало
# бы кросс-источниковый дедуп, поэтому трактуем как дефолт-минимум 1 день.
MIN_DEDUP_WINDOW_DAYS = 1
# Верхняя граница окна. Дефолт — 3 дня; кросс-источниковые дубли старше месяца
# практически не встречаются (retention/доставка работают на порядок уже окна).
# Опечатка вроде window_days: 300 не должна сканировать заголовки за год.
MAX_DEDUP_WINDOW_DAYS = 30


def _cfg_bool(config: dict, key: str, default: bool) -> bool:
    """bool-поле dedup.*: bool как есть, int — как 0/1, строка — только true/1/yes/on."""
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    # YAML/JSON-конфиг может содержать 0/1 для булева поля. Раньше `cfg.get()`
    # трактовал 0 как falsy; сохраняем это поведение без warning, иначе
    # cross_source: 0 (дефолт True) молча ВКЛЮЧАЛ бы кросс-дедуп + спамил warning.
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off", ""):
            return False
    logger.warning(
        "dedup.%s=%r — ожидалось true/false, используется дефолт %r",
        key, value, default,
    )
    return default


def _cfg_float(config: dict, key: str, default: float) -> float:
    """float-поле dedup.*: невалидное значение → warning + дефолт."""
    value = config.get(key, default)
    if isinstance(value, bool):
        logger.warning(
            "dedup.%s=%r — ожидалось число, используется дефолт %r",
            key, value, default,
        )
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning(
            "dedup.%s=%r — ожидалось число, используется дефолт %r",
            key, value, default,
        )
        return default
    if not math.isfinite(parsed):
        # NaN/inf из конфига (например "nan") — невалидны для порогов.
        logger.warning(
            "dedup.%s=%r — не конечное число, используется дефолт %r",
            key, value, default,
        )
        return default
    return parsed


def _cfg_int(config: dict, key: str, default: int) -> int:
    """int-поле dedup.*: невалидное значение → warning + дефолт.

    Нецелый float (2.9) не усекается молча: warning + округление ВНИЗ.
    Консистентно с _cfg_float: разбираем через float (принимаем "3", "3.0"),
    NaN/inf → дефолт. Но в отличие от порогов (где неверное значение молча
    ломает дедуп), у окна дней намерение очевидно — floor безопаснее дефолта,
    а warning сохраняет наблюдаемость опечатки.
    """
    value = config.get(key, default)
    if isinstance(value, bool):
        logger.warning(
            "dedup.%s=%r — ожидалось целое число, используется дефолт %r",
            key, value, default,
        )
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning(
            "dedup.%s=%r — ожидалось целое число, используется дефолт %r",
            key, value, default,
        )
        return default
    if not math.isfinite(parsed):
        logger.warning(
            "dedup.%s=%r — не конечное число, используется дефолт %r",
            key, value, default,
        )
        return default
    floored = int(math.floor(parsed))
    if floored != parsed:
        logger.warning(
            "dedup.%s=%r — нецелое число, округлено вниз до %d",
            key, value, floored,
        )
    return floored


def _clamp_unit_interval(key: str, value: float) -> float:
    """Кламп порога в [0,1] с warning'ами (Minor-фикс: кламп был молчаливым).

    Граничные значения (0.0 / 1.0) означают «критерий фактически отключён»:
    0 — совпадает всё, 1 — только побайтово идентичные строки. Логируем это
    отдельным warning'ом, чтобы оператор видел причину «дедуп не работает».
    """
    clamped = max(0.0, min(1.0, value))
    if clamped != value:
        logger.warning(
            "dedup.%s=%r вне диапазона [0,1] — кламплено в %r",
            key, value, clamped,
        )
    if clamped <= 0.0 or clamped >= 1.0:
        logger.warning(
            "dedup.%s=%r — fuzzy-дедуп по этому критерию фактически "
            "отключён (граничное значение)",
            key, clamped,
        )
    return clamped


# ─── Предфильтр для fuzzy-дедупа ──────────────────────────────────────────────
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
    r"""
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
    r"""
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
      (d) быстрый доменный дедуп: тот же cluster_key (тема+день) за последние
          2 часа, независимо от source
      (e) [опц.] нечёткое сравнение (SequenceMatcher) кросс-источник,
          только для осмысленных заголовков, ТОЛЬКО когда cross_source=true
          (fuzzy=true И cross_source=true)

    Формула fuzzy-дубля (guard'ы — до SequenceMatcher):
        дубль ⟺ (числа совпадают) И (Jaccard ≥ token_threshold) И (ratio ≥ threshold)

    При нахождении дубля уровня (c) или (d) — item сохраняется в items
    со state='skipped' (URL-дедуп в следующих циклах + не попадает в delivery).

    Возвращает (new_items, total_count, dup_count).
    """
    cfg = dedup_config if dedup_config is not None else {}
    # P1-фикс: защищённый разбор dedup.* — опечатка в config.yaml не должна
    # ронять filter_new (иначе цикл обрывается до save_many → потери новостей).
    cross_source = _cfg_bool(cfg, "cross_source", True)
    fuzzy = _cfg_bool(cfg, "fuzzy", True)
    # NOTE: fuzzy выполняется ТОЛЬКО при cross_source=true (кросс-источниковый контекст).
    # При cross_source=false fuzzy игнорируется — это сохраняет семантику per-source
    # дедупа (иначе возникает неочевидное поведение: разные source, но fuzzy всё склеивает).
    if cross_source:
        fuzzy_enabled = fuzzy
    else:
        fuzzy_enabled = False  # явно выключаем при cross_source=false

    threshold = _cfg_float(cfg, "threshold", 0.95)
    token_threshold = _cfg_float(cfg, "token_threshold", 0.8)
    window_days = _cfg_int(cfg, "window_days", 3)
    if window_days < MIN_DEDUP_WINDOW_DAYS:
        logger.warning(
            "dedup.window_days=%r вне диапазона — используется %d",
            window_days, MIN_DEDUP_WINDOW_DAYS,
        )
        window_days = MIN_DEDUP_WINDOW_DAYS
    elif window_days > MAX_DEDUP_WINDOW_DAYS:
        logger.warning(
            "dedup.window_days=%r больше максимума — используется %d",
            window_days, MAX_DEDUP_WINDOW_DAYS,
        )
        window_days = MAX_DEDUP_WINDOW_DAYS

    # Лимит выборки заголовков за окно. None/невалидное/отрицательное → 2000
    # (дефолт), 0 в конфиге → без лимита (только окно). Жёсткий LIMIT 500 молча
    # усекал кросс-source дедуп при >500 заголовках за окно.
    # Отрицательное значение (опечатка вроде -5) НЕ превращается в 0: это молча
    # снимало бы страховку от неограниченной выборки, поэтому → дефолт 2000.
    max_seen_titles = cfg.get("max_seen_titles", 2000)
    if isinstance(max_seen_titles, bool):
        max_seen_titles = 2000
    else:
        # Major-фикс: YAML-идиома `max_seen_titles: .inf` («без лимита») давала
        # float('inf'), а int(inf) бросает OverflowError → filter_new падал →
        # цикл обрывался ДО save_many (потеря новостей). Не-finite float
        # обрабатываем ДО int(); inf трактуем как «без лимита» = 0 (ровно то,
        # что уже означает 0 в контракте), nan — как опечатку → дефолт 2000.
        _numeric = None
        if isinstance(max_seen_titles, float):
            _numeric = max_seen_titles
        elif isinstance(max_seen_titles, str):
            try:
                _numeric = float(max_seen_titles)
            except ValueError:
                _numeric = None
        if _numeric is not None and not math.isfinite(_numeric):
            if _numeric > 0:  # +inf (для -inf ветка < 0 ниже ведёт в дефолт)
                logger.warning(
                    "dedup.max_seen_titles=%r — трактуется как «без лимита» (0)",
                    max_seen_titles,
                )
                max_seen_titles = 0
            else:
                logger.warning(
                    "dedup.max_seen_titles=%r — не конечное число, "
                    "используется 2000",
                    max_seen_titles,
                )
                max_seen_titles = 2000
        else:
            try:
                max_seen_titles = int(max_seen_titles)
            except (TypeError, ValueError, OverflowError):
                logger.warning(
                    "dedup.max_seen_titles=%r — ожидалось целое число, "
                    "используется 2000",
                    max_seen_titles,
                )
                max_seen_titles = 2000
    if max_seen_titles < 0:
        max_seen_titles = 2000

    # clamp thresholds (с warning — молчаливый кламп скрывал опечатку)
    threshold = _clamp_unit_interval("threshold", threshold)
    token_threshold = _clamp_unit_interval("token_threshold", token_threshold)

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
    # Флаг «выборка упёрлась в лимит» — warning один раз за цикл, не на item.
    _recent_limit_hit = False

    # Предварительно собираем множество source'ов, которые встретятся
    # в цикле, чтобы запросить их ДО начала основного цикла.
    # Это позволяет сделать все запросы заранее, а не по требованию.
    if cross_source or fuzzy_enabled:
        for item in items:
            if _is_meaningful_title(item):
                src = item.source
                if src not in _recent_cache:
                    rows = storage.get_recent_normalized_titles(
                        window_days=window_days,
                        limit=max_seen_titles,
                        exclude_source=src,
                    )
                    _recent_cache[src] = rows
                    if max_seen_titles and len(rows) >= max_seen_titles:
                        _recent_limit_hit = True

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

        # ── Быстрый доменный дедуп: тот же cluster_key за последние 2 часа ──
        if _domain_recent_duplicate(item, storage):
            storage.save_skipped(item)
            dup_count += 1
            logger.debug(
                "Доменный дубль (recent): source=%s title=%r",
                item.source, item.title[:60],
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

    if _recent_limit_hit:
        logger.warning(
            "Выборка дедупа усечена лимитом: проверено не более %d заголовков "
            "за окно (dedup.max_seen_titles; 0 = без лимита)",
            max_seen_titles,
        )
    if cross_source_skipped:
        logger.info("Дедуп: отфильтровано %d cross-source дублей", cross_source_skipped)
    if fuzzy_skipped:
        logger.info("Дедуп: отфильтровано %d fuzzy-дублей", fuzzy_skipped)

    return new_items, len(items), dup_count