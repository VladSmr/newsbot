r"""
Дедупликация: обёртка над Storage.

Внутри-доменный дедуп (Задача 2.4):
  Если один и тот же cluster_key (тема+день) за последние 2 часа,
  независимо от source — пропускаем новый item как потенциальный дубль.
  Выполняется ДО fuzzy-дедупа (быстрая отсечка).
  Числовой guard: stem_hash не содержит цифр (normalize() их вырезает),
  поэтому совпадение cluster_key дополнительно сверяется по мультимножеству
  чисел заголовков — «25 bps» и «50 bps» больше не схлопываются в дубль.

  Major#10-фикс: prefetch-оптимизация. Раньше каждый item делал свой SQL-запрос
  с LIKE '%' || suffix, что вызывало полнотабличный скан на каждый item
  (LIKE с leading wildcard не использует индекс idx_items_cluster_key_fetched).
  Теперь: ОДИН раз за вызов filter_new префетчим все (cluster_key, title)
  за 2-часового окна одним запросом (индекс idx_items_fetched_at v16),
  результат — Python-dict для O(1) lookup. Сложность: O(N) запросов → O(1).

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
from typing import Dict, List, Optional, Set, Tuple

from .collector import NewsItem, _strip_source_prefix
from .storage import Storage, MIN_TITLE_LEN

logger = logging.getLogger("newsbot.dedup")


# ─── Внутри-доменный дедуп (Задача 2.4) ──────────────────────────────────────




def _domain_recent_duplicate_legacy(
    item: NewsItem,
    prefetch: Optional[Dict[str, Tuple[str, str]]],
) -> bool:
    """
    NOT IN PROD PATH — не используется filter_new; НЕ содержит first-2
    и within-cycle числовой guard. Не подключать обратно без переноса
    этих проверок.

    Быстрый дедуп по cluster_key: если ОДИН cluster_key (тема+день)
    уже был за последние 2 часа — блокируем. Разные темы от одного
    источника НЕ блокируются.

    Major#10-фикс (prefetch-паттерн): принимает prefetch-dict вместо
    обращения к БД напрямую. Prefetch выполняется ОДИН раз за вызов
    filter_new через storage.get_recent_for_dedup(): один SQL-запрос
    загружает все (cluster_key_suffix, title) за 2-часового окна, затем
    O(1) lookup per item вместо N SQL-запросов (каждый с LIKE-сканом).

    M2-фикс (сохранён): сравниваем по суффиксу stem|day (без локационного
    префикса). Инвариант guard: ловит «та же тема + тот же день»,
    независимо от пересобранного ключа в БД.

    NOTE: prefetch=None трактуется как «нет данных» → fail-open (не дубль).
    Это защищает filter_new от падения, если prefetch-получение данных
    не удалось (ошибка БД и т.п.).

    Числовой guard (сохранён): «25 bps» vs «50 bps» — разные числа →
    НЕ дубль (идёт дальше по конвейеру).
    """
    cluster_key = getattr(item, 'cluster_key', None)
    if not cluster_key:
        return False
    if prefetch is None:
        # Fail-open: prefetch недоступен → пропускаем (не дубль).
        return False

    # Числа нового item: fail-open при ошибке разбора.
    item_numbers = _safe_number_multiset(getattr(item, 'title', None))

    # Определяем lookup-ключ в prefetch-dict:
    # новый формат 'loc|stem|day' → суффикс 'stem|day';
    # старый формат 'cb_rate_2024' → полный ключ.
    if '|' in cluster_key:
        lookup_key = cluster_key.split('|', 1)[1]
    else:
        lookup_key = cluster_key

    candidates = prefetch.get(lookup_key)
    if candidates is None or not candidates:
        return False

    if item_numbers is None:
        # Числа нового item не разобрались → прежнее поведение:
        # cluster_key совпал ⇒ дубль (без числовой сверки).
        return True

    # Хотя бы один candidate совпадает по числам → дубль (OR semantics).
    # Например: два item с одинаковым суффиксом 'stem123|day', но разными
    # числами (70 000 vs 80 000). Если новый item с числом 70000 —
    # совпадает с первым кандидатом → дубль; с 90000 — не совпадает ни с одним.
    for cand_title, _ in candidates:
        cand_numbers = _safe_number_multiset(cand_title)
        if cand_numbers is None or cand_numbers == item_numbers:
            return True
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


def _first_two_content_tokens(title: str) -> tuple:
    """
    Первые два слова заголовка (сырые, без стемминга).
    Используется для Major-2: быстрая проверка «та же тема» без стемминга.
    Одинаковые first-2 → дубли; разные → разные темы.

    Фильтрация стоп-слов и слов ≤2 символов НЕ применяется: именно
    короткие слова часто несут ключевую информацию и должны совпадать
    для склейки. Пример: «EU and US reach deal» даст ('eu', 'and'), где
    'and' — информативный коннектор в данном контексте. Без фильтрации
    сохраняется предсказуемость: first-2 = ровно то, что видит читатель
    в начале заголовка, без эвристик.
    """
    tokens = [w.strip().lower() for w in (title or '').split()]
    return tuple(tokens[:2])


def _number_multiset(text: str) -> Counter:
    r"""
    Мультимножество числовых токенов текста. Жёсткий guard: разные
    мультимножества → НЕ дубль. Counter, а не set: "70 70" и "70" — РАЗНЫЕ
    мультимножества (2 vs 1).

    Регекс `\d+(?:[.,]\d+)?` чувствителен к тому, ЧТО подано на вход — два
    call-site с разной семантикой:
      • fuzzy-дедуп (`_is_fuzzy_similar`) — НОРМАЛИЗОВАННЫЙ текст
        (`NewsItem.title_normalized()`), где [.,] и прочая пунктуация уже
        вырезаны, поэтому регекс фактически ловит только `\d+`;
      • доменный 2h-guard (`_domain_recent_duplicate`) — СЫРОЙ заголовок
        из `items.title`, поэтому десятичные сохраняются одним токеном
        («2.5», «2,5») и не путаются с двумя числами «2 5».
    Семантика намеренно не выравнивается: на сыром тексте guard строже
    (десятичная дробь — отдельный токен), а строже = безопаснее (меньше
    ложных склеек; расхождение чисел лишь пропускает item дальше по конвейеру).
    """
    return Counter(_NUMBER_RE.findall(text))


def _safe_number_multiset(text) -> Optional[Counter]:
    """
    Fail-open обёртка над _number_multiset для данных из внешнего кода.

    Возвращает None, если числа посчитать не удалось (не-str title и т.п.):
    вызывающий код трактует None как «сверка невозможна» и деградирует до
    прежнего поведения. Исключение НЕ пробрасывается — иначе filter_new
    оборвался бы до save_many (потеря новостей в цикле).
    """
    if text is None:
        # NULL/отсутствующий title = пустое множество чисел (как '').
        return Counter()
    try:
        if not isinstance(text, str):
            text = str(text)
        return _number_multiset(text)
    except Exception as e:
        logger.warning(
            "Доменный дедуп: не удалось разобрать числа заголовка (type=%s): %s",
            type(text).__name__, e,
        )
        return None


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


def _is_containment_duplicate_check(
    shorter: str,
    longer: str,
    nums_shorter: Counter,
    nums_longer: Counter,
    meaningful_set_shorter: Set[str],
    meaningful_set_longer: Set[str],
    token_threshold: float,
) -> bool:
    """
    Внутренний helper containment-проверки.
    Major-3: принимает предвычисленные nums и token_sets, чтобы не
    пересчитывать их в _is_fuzzy_similar (где они уже известны).

    Контракт (вызывающий код гарантирует):
      1. shorter и longer — нормализованные заголовки.
      2. len(shorter) <= len(longer).
      3. Оба ≥ 30 символов.
      4. nums_shorter == nums_longer (числовой guard уже пройден).
    """
    # Числовой guard (повторная проверка — для безопасности).
    if nums_shorter != nums_longer:
        return False

    # Token guard: Jaccard значащих токенов.
    if meaningful_set_shorter and meaningful_set_longer:
        intersection = len(meaningful_set_shorter & meaningful_set_longer)
        union = len(meaningful_set_shorter | meaningful_set_longer)
        if (intersection / union) < token_threshold:
            return False

    # Containment по границе слова.
    idx = 0
    while True:
        pos = longer.find(shorter, idx)
        if pos == -1:
            break
        left_ok = (pos == 0) or longer[pos - 1].isspace()
        right_ok = (pos + len(shorter) >= len(longer)) or longer[pos + len(shorter)].isspace()
        if left_ok and right_ok:
            return True
        idx = pos + 1
    return False


def _token_set(text: str) -> Set[str]:
    """Токен-сет нормализованного текста (набор уникальных слов)."""
    return set(text.split())


def _quick_filter(
    norm: str,
    candidates: List[Tuple],
    precomputed_token_sets: Optional[Dict[str, Set[str]]] = None,
) -> Tuple[List[Tuple], bool]:
    """
    Предфильтр по длине и токен-сету.
    Возвращает (subset, limit_hit) — подмножество candidates и флаг усечения.
    O(N) без SequenceMatcher — быстрый отсев заведомо непохожих.

    MINOR-3: warning об усечении вынесен на уровень filter_new (1 раз за цикл).
    Major-3: если precomputed_token_sets передан — переиспользует
    предвычисленные токен-сеты вместо пересчёта на каждый candidate.
    """
    norm_tokens = _token_set(norm)
    if not norm_tokens:
        return [], False

    norm_words = norm.split()
    norm_token_count = len(norm_tokens)

    filtered: List[Tuple] = []
    limit_hit = False
    for title_hash, candidate_norm, *_ in candidates:
        if not candidate_norm:
            continue
        cand_words = candidate_norm.split()

        # (1) фильтр по длине
        cand_len_range = (len(cand_words) - MAX_LEN_DIFF, len(cand_words) + MAX_LEN_DIFF)
        if not (cand_len_range[0] <= len(norm_words) <= cand_len_range[1]):
            continue

        # (2) фильтр по токен-сету: используем предвычисленный или считаем
        if precomputed_token_sets is not None:
            cand_tokens = precomputed_token_sets.get(candidate_norm)
        else:
            cand_tokens = None
        if cand_tokens is None:
            cand_tokens = _token_set(candidate_norm)
        if not cand_tokens:
            continue
        intersection = len(norm_tokens & cand_tokens)
        token_ratio = intersection / min(norm_token_count, len(cand_tokens))
        if token_ratio < MIN_TOKEN_INTERSECTION:
            continue

        filtered.append((title_hash, candidate_norm))
        if len(filtered) > MAX_FUZZY_CANDIDATES:
            limit_hit = True
            filtered = filtered[:MAX_FUZZY_CANDIDATES]
            break

    return filtered, limit_hit


def is_containment_duplicate(title_a: str, title_b: str, token_threshold: float = 0.8) -> bool:
    r"""
    Containment-дедуп: один нормализованный заголовок полностью входит в другой
    по границе слова, при этом числовой guard и token-guard пройдены.

    Публичная функция — сигнатура стабильна, используется внешними модулями.

    Контракт (все проверки обязательны, без них функция НЕ безопасна):
      1. Оба заголовка нормализованы (lowercase, без пунктуации, без source-префикса).
      2. Длина обоих ≥ 30 символов после нормализации — защита от ложных склеек
         коротких заголовков («EU approves» vs «EU approves law» — не дубль,
         даже если один входит в другой; порог 30 симв. выбран эмпирически).
      3. Containment по границе слова: `\b` в начале и конце входящего,
         чтобы «moon» не совпал с «moonlight», а «Two killed … Voronezh»
         совпал с «Two killed … Voronezh region …».
      4. Числовой guard (мультимножество цифр) совпадает — защита от ложной
         склейки «2 killed» vs «3 killed», «70 bps» vs «80 bps» и т.п.
      5. Token Jaccard ≥ token_threshold — заголовки описывают одну сущность.
         Guard пропускает при отсутствии значащих токенов.

    NOTE: ratio НЕ проверяется — containment сам по себе достаточно сильный сигнал.
    Проверка словлена до ratio, чтобы поймать пары типа
    «Two killed in Ukrainian attack on Voronezh — ТАСС» /
    «Two killed in Ukrainian attack on Voronezh — РИА»,
    где SequenceMatcher ratio ≈ 0.935 < 0.95 и пара отклонялась бы.

    Args:
        title_a: нормализованный заголовок ( уже через title_normalized() ).
        title_b: нормализованный заголовок.
        token_threshold: порог Jaccard (дефолт 0.8, соответствует конфигу dedup).
    Returns:
        True — дубль (один заголовок целиком входит в другой, guard'ы пройдены).
        False — не дубль или guard отклонил.
    """
    if not title_a or not title_b:
        return False

    a_len = len(title_a)
    b_len = len(title_b)

    # Оба ≥ 30 символов (защита от ложных склеек на коротких заголовках)
    if a_len < 30 or b_len < 30:
        return False

    # ── (1) Числовой guard — мультимножество ─────────────────────────
    nums_a = _number_multiset(title_a)
    nums_b = _number_multiset(title_b)
    if nums_a != nums_b:
        return False

    # ── (2) Token guard — Jaccard ───────────────────────────────────
    if not _token_jaccard(title_a, title_b, token_threshold):
        return False

    # ── (3) Containment по границе слова ────────────────────────────
    if a_len <= b_len:
        shorter, longer = title_a, title_b
    else:
        shorter, longer = title_b, title_a

    # Делегируем во внутренний helper: вычисляем token-сеты здесь,
    # чтобы is_containment_duplicate остался self-contained (вызывается извне).
    meaningful_shorter = _meaningful_tokens(shorter)
    meaningful_longer = _meaningful_tokens(longer)
    return _is_containment_duplicate_check(
        shorter, longer,
        nums_a, nums_b,
        meaningful_shorter, meaningful_longer,
        token_threshold,
    )


def _is_fuzzy_similar(
    a: str, b: str, threshold: float, token_threshold: float,
    _precomputed: Optional[dict] = None,
) -> bool:
    r"""
    Сравнивает два нормализованных заголовка через SequenceMatcher
    с дополнительной защитой от ложных склеек.

    Формула «дубль» (guard'ы ДО SequenceMatcher — дешёвые раньше дорогих):
        числа совпадают (мультимножество)  — жёсткий запрет при расхождении
        И token_Jaccard ≥ token_threshold  — значимые токены похожи
        И (containment  — OR — ratio ≥ threshold)

    Containment-проверка выполняется ДО SequenceMatcher: если один заголовок
    целиком входит в другой по границе слова (оба ≥30 симв., числовой guard
    и Jaccard пройдены), пара считается дублем БЕЗ вычисления ratio.

    NOTE: числовой guard работает на нормализованном тексте. Поскольку
    title_normalized() удаляет [^\w\s], десятичные точки, запятые, знаки
    и % уже вырезаны. Мультимножество (Counter) отличает «70 70» от «70».

    Args:
        a: нормализованный заголовок нового item
        b: нормализованный заголовок кандидата из БД
        threshold: порог SequenceMatcher ratio (дефолт 0.95)
        token_threshold: порог Jaccard значащих токенов (дефолт 0.8)
        _precomputed: optional dict with precomputed values from filter_new
                      (nums_a). Used to avoid recomputing item numbers
                      on every candidate. All other values computed fresh.
    """
    if not a or not b:
        return False

    # ── (1) Числовой guard — мультимножество (до SequenceMatcher) ─────
    # nums_a: предвычислен (один раз за item); nums_b: вычисляется fresh
    # для каждого candidate (другой candidate → другие nums).
    if _precomputed is not None and 'nums_a' in _precomputed:
        nums_a = _precomputed['nums_a']
    else:
        nums_a = _number_multiset(a)
    nums_b = _number_multiset(b)
    if nums_a != nums_b:
        return False

    # ── (2) Token guard — Jaccard значащих токенов ────────────────────
    meaningful_a = _meaningful_tokens(a)
    meaningful_b = _meaningful_tokens(b)
    if meaningful_a and meaningful_b:
        intersection = len(meaningful_a & meaningful_b)
        union = len(meaningful_a | meaningful_b)
        if (intersection / union) < token_threshold:
            return False
    # empty sets: guard пропускает

    # ── (3) Containment-правило (без ratio), используя precomputed ───
    a_len = len(a)
    b_len = len(b)
    if a_len >= 30 and b_len >= 30:
        if a_len <= b_len:
            shorter, longer = a, b
        else:
            shorter, longer = b, a
        containment_result = _is_containment_duplicate_check(
            shorter, longer,
            nums_a, nums_b,
            meaningful_a, meaningful_b,
            token_threshold,
        )
        if containment_result:
            return True

    # ── (4) SequenceMatcher ratio (дорогой — последним) ───────────────
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
    # MINOR-3: флаг усечения fuzzy-пула — warning один раз за цикл, не на item.
    _fuzzy_limit_hit = False
    # Major-3: кэш предвычисленных токен-сетов кандидатов по source.
    # Major-2: within-cycle seen first-2-tokens по cluster_key_suffix.
    # Строится лениво при первом fuzzy-вызове для каждого source.
    _token_sets_cache: Dict[str, Dict[str, Set[str]]] = {}
    # {lookup_key: {first2_tuple → nums_counter_or_None}}
    # last-write-wins: если третий item пришёл с теми же first-2,
    # но числами первого item'а в том же цикле — он НЕ распознается как дубль.
    # Это осознанный компромисс (не баг): экономия памяти и простота реализации.
    _domain_cycle_seen: Dict[str, Dict[tuple, Optional[Counter]]] = {}

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

    # Major#10-фикс: prefetch всех cluster_key для _domain_recent_duplicate.
    # ОДИН запрос вместо N. Ошибка → prefetch=None → fail-open guard (не дубль).
    _prefetch_cache: Optional[Dict[str, Tuple[str, str]]] = None
    # MINOR-1: предвычисленные (first2, nums) для каждого candidate по title.
    # Строится ОДИН раз за вызов filter_new, переиспользуется для всех items.
    _prefetch_metadata: Dict[str, Dict[str, Tuple[tuple, Counter]]] = {}
    # MAJOR-prefetch-фикс: prefetch_window_hours из конфига (ранее жёстко 2).
    # dedup.py читает hours из config dedup.prefetch_window_hours, дефолт 2.
    prefetch_hours = 2  # обратная совместимость
    if dedup_config and isinstance(dedup_config, dict):
        raw = dedup_config.get("prefetch_window_hours")
        if raw is not None:
            try:
                f = float(raw)
                if f == int(f) and 1 <= int(f) <= 48:
                    prefetch_hours = int(f)
                else:
                    logger.warning(
                        "dedup.prefetch_window_hours=%r вне диапазона [1..48], "
                        "используется дефолт 2.", raw,
                    )
            except (TypeError, ValueError):
                logger.warning(
                    "dedup.prefetch_window_hours=%r — ожидалось целое, "
                    "используется дефолт 2.", raw,
                )
    try:
        _prefetch_cache = storage.get_recent_for_dedup(hours=prefetch_hours)
    except Exception as e:
        logger.warning(
            "Prefetch для доменного дедупа не удался: %s. "
            "Доменный guard пропускает все items (fail-open).", e,
        )
        _prefetch_cache = None
    # MINOR-4: отдельный try/except — metadata ошибка не влияет на prefetch-кеш.
    if _prefetch_cache:
        try:
            for lookup_key, candidates in _prefetch_cache.items():
                meta: Dict[str, Tuple[tuple, Counter]] = {}
                for c_t, _ in candidates:
                    if c_t and c_t not in meta:
                        raw = _strip_source_prefix(c_t)
                        meta[c_t] = (
                            _first_two_content_tokens(raw),
                            _safe_number_multiset(c_t),
                        )
                _prefetch_metadata[lookup_key] = meta
        except Exception as e:
            logger.warning(
                "Prefetch metadata build failed: %s. "
                "Domain guard uses prefetch cache without precomputed metadata (fail-open).", e,
            )
            _prefetch_metadata = {}

    for item in items:
        # (a) точный URL в БД
        if storage.is_duplicate(item):
            dup_count += 1
            logger.debug("URL-дубликат пропущен: %s", item.url[:80])
            continue

        normalized = item.title_normalized()
        # Для Major-2 (domain-дедуп) — вычисляем один раз на item.
        item_numbers = _safe_number_multiset(getattr(item, 'title', None))
        item_first2 = _first_two_content_tokens(
            _strip_source_prefix(getattr(item, 'title', None) or '')
        )

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
        # Major#10: prefetch-кеш вместо per-item SQL-запроса.
        # Major-2: защита от склейки противоположных по смыслу заголовков.
        # normalize() сортирует слова, поэтому cluster_key нечувствителен
        # к порядку → «Russia strikes Ukraine» и «Ukraine strikes Russia» дают
        # один ключ. Подтверждающая дешёвая проверка на first-2-content-tokens
        # (без стоп-слов, без стемминга): одинаковые первые два информативных
        # слова → дубли (та же тема); разные → разные темы (не склеиваем).
        #
        # Проверяется ДВА источника «уже виденного»:
        #   (a) prefetch cache — существующие items из БД за 2 часа
        #   (b) _domain_cycle_seen — items, обработанные в этом цикле (within-cycle)
        is_domain_dup = False
        if _prefetch_cache is not None:
            item_cluster_key = getattr(item, 'cluster_key', None)
            if item_cluster_key:
                if '|' in item_cluster_key:
                    item_lookup_key = item_cluster_key.split('|', 1)[1]
                else:
                    item_lookup_key = item_cluster_key
                item_candidates = _prefetch_cache.get(item_lookup_key)
                # MINOR-1: используем предвычисленный кэш метаданных candidates.
                # (item_first2 уже вычислен в начале цикла.)

                def _matches_any_candidate(first2_tokens, candidates_list, item_nums_check, meta_cache):
                    """Есть ли candidate с теми же first-2 токенами и (опц.) числами?

                    MINOR-1: first2 и nums берутся из предвычисленного meta_cache
                    (построен один раз за вызов filter_new), а не пересчитываются.
                    """
                    for c_t, _ in candidates_list:
                        entry = meta_cache.get(c_t)
                        if entry is None:
                            continue
                        c_first2, c_nums = entry
                        if c_first2 == first2_tokens:
                            if item_nums_check is None:
                                # Числа не доступны — match без числовой сверки.
                                return True
                            if c_nums is None or c_nums == item_nums_check:
                                return True
                    return False

                if item_candidates:
                    if item_numbers is None:
                        # Числа не разобрались — прежнее поведение: cluster_key
                        # совпал, числа не проверить → дубль (без Major-2).
                        is_domain_dup = True
                    else:
                        # (a) Совпадает ли с каким-то существующим по first-2 + numbers?
                        # MINOR-1: используем предвычисленный кэш метаданных.
                        meta_cache = _prefetch_metadata.get(item_lookup_key, {})
                        if _matches_any_candidate(
                                item_first2, item_candidates, item_numbers, meta_cache):
                            is_domain_dup = True
                        else:
                            # (b) Within-cycle: был ли уже обработан item с теми же
                            # first-2 токенами И числами в этом цикле (lookup_key)?
                            seen_for_key = _domain_cycle_seen.get(item_lookup_key, {})
                            item_nums_key = item_numbers if item_numbers is not None else None
                            if item_first2 in seen_for_key:
                                seen_nums = seen_for_key[item_first2]
                                if seen_nums is None or seen_nums == item_nums_key:
                                    is_domain_dup = True
                else:
                    # Нет существующих кандидатов с этим cluster_key.
                    # (b) Within-cycle: был ли уже обработан item с теми же
                    # first-2 + numbers?
                    seen_for_key = _domain_cycle_seen.get(item_lookup_key, {})
                    item_nums_key = item_numbers if item_numbers is not None else None
                    if item_first2 in seen_for_key:
                        seen_nums = seen_for_key[item_first2]
                        if seen_nums is None or seen_nums == item_nums_key:
                            is_domain_dup = True
        if is_domain_dup:
            storage.save_skipped(item)
            dup_count += 1
            logger.debug(
                "Доменный дубль (recent): source=%s title=%r",
                item.source, item.title[:60],
            )
            continue

        # Регистрируем first-2 в within-cycle cache (Domain-2).
        # Вызывается ПОСЛЕ domain dedup (is_domain_dup=False) и ДО fuzzy dedup:
        # item1 регистрируется сразу, item2 сразу видит её и блокируется.
        if _prefetch_cache is not None:
            item_cluster_key_reg = getattr(item, 'cluster_key', None)
            if item_cluster_key_reg:
                if '|' in item_cluster_key_reg:
                    lookup_key_reg = item_cluster_key_reg.split('|', 1)[1]
                else:
                    lookup_key_reg = item_cluster_key_reg
                if lookup_key_reg not in _domain_cycle_seen:
                    _domain_cycle_seen[lookup_key_reg] = {}
                _domain_cycle_seen[lookup_key_reg][item_first2] = item_numbers

        # ── Уровень 2: fuzzy-дедуп (только при cross_source=true) ──
        fuzzy_duplicate = False
        if fuzzy_enabled and recent:
            # Major-3: предвычисляем токен-сеты кандидатов один раз за цикл
            # (для всех items одного source — shared). Строим один раз,
            # затем переиспользуем в _quick_filter и _is_fuzzy_similar.
            if item.source not in _token_sets_cache:
                _token_sets_cache[item.source] = {
                    cand_norm: _token_set(cand_norm)
                    for _, cand_norm, *_ in _recent_cache.get(item.source, [])
                    if cand_norm
                }
            cand_token_sets = _token_sets_cache[item.source]

            # предфильтр: быстрый отсев по длине и токен-сету
            filtered, limit_hit = _quick_filter(
                normalized, recent, precomputed_token_sets=cand_token_sets)
            if limit_hit:
                _fuzzy_limit_hit = True
            # Предвычисляем nums и meaningful для item (один раз за item).
            item_nums = _number_multiset(normalized)
            for _, cand_norm, *_ in filtered:
                precomputed = {'nums_a': item_nums}
                if _is_fuzzy_similar(
                    normalized, cand_norm, threshold, token_threshold,
                    _precomputed=precomputed,
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
    if _fuzzy_limit_hit:
        logger.warning(
            "Fuzzy-пул усечён: лимит %d кандидатов. "
            "Увеличьте dedup.max_seen_titles для полного покрытия.",
            MAX_FUZZY_CANDIDATES,
        )
    if cross_source_skipped:
        logger.info("Дедуп: отфильтровано %d cross-source дублей", cross_source_skipped)
    if fuzzy_skipped:
        logger.info("Дедуп: отфильтровано %d fuzzy-дублей", fuzzy_skipped)

    return new_items, len(items), dup_count