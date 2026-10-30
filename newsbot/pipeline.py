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


import hashlib
import logging
import math
import re
import time as _time
from datetime import datetime, timedelta, timezone as tz
from email.utils import parsedate_to_datetime as _parsedate_to_dt
from typing import Dict, List, Optional, Set, Tuple

from .collector import FeedCollector, FeedResult, NewsItem
from .dedup import filter_new, is_containment_duplicate
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


def _local_now() -> datetime:
    """
    Returns current UTC time converted to local timezone.

    Extracted to a function so tests can patch it for TZ-independent testing.
    Windows pytest can't use time.tzset(), so instead we patch this function.
    """
    return datetime.now(tz.utc).astimezone()


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


# Дефолтные слоты расписания дайджеста. Синхронизировано с
# config.DEFAULT_CONFIG["digest"]["times"] = ["08:00", "20:00"].
_DEFAULT_DIGEST_TIMES = ("08:00", "20:00")


def _normalize_digest_times(raw) -> List[tuple]:
    """
    Нормализует digest.times из YAML в список (hour, minute).

    Раньше валидации не было, и любой мусор в конфиге убивал расписание:
      (а) ["9:00", "21:00"] — int("9:") → ValueError перехвачен лишь
          частично: слот молча выключался (1 дайджест в сутки вместо 2);
      (б) [9:00] без кавычек — YAML sexagesimal → int 540 → `t[:2]` →
          TypeError ВНЕ перехвата → _maybe_run_digest падал каждый цикл,
          дайджест не запускался ВООБЩЕ;
      (в) "08:00" строкой — итерация по символам ('0','8',':') → мусорные
          слоты/TypeError.

    Правила (все аномалии — warning, без исключений):
      — не список/кортеж (в т.ч. строка) → warning + дефолт из config;
      — None (ключа нет) → дефолт без warning;
      — явный пустой список → расписание осознанно выключено → [];
      — элемент не str (напр. int 540 из YAML sexagesimal) → warning + skip;
      — str не парсится как HH:MM → warning + skip;
      — "9:00"/" 9:00 " → нормализация в "09:00" с warning (однозначно
        читаемое значение поддерживаем, а не выключаем слот молча);
      — если после фильтрации не осталось ни одного слота, а конфиг был
        непустой → warning + дефолт (иначе одна опечатка навсегда
        отключала бы дайджест).

    Парсинг через time.strptime: та же строгость, что у
    datetime.strptime("%H:%M"), но не завязан на module-level
    `datetime` (его подменяют в тестах расписания).
    """
    default_slots = [
        (int(t[:2]), int(t[3:])) for t in _DEFAULT_DIGEST_TIMES
    ]
    if raw is None:
        return default_slots
    if not isinstance(raw, (list, tuple)):
        logger.warning(
            "digest.times: ожидался список строк 'HH:MM', получено %r (%s) — "
            "использую дефолт %s",
            raw, type(raw).__name__, list(_DEFAULT_DIGEST_TIMES),
        )
        return default_slots
    if not raw:
        # Явный пустой список — штатное «расписание выключено» (без шума).
        return []

    slots: List[tuple] = []
    for t in raw:
        if not isinstance(t, str):
            logger.warning(
                "digest.times: элемент %r не строка (%s) — слот пропущен "
                "(для YAML берите время в кавычках: \"09:00\")",
                t, type(t).__name__,
            )
            continue
        candidate = t.strip()
        try:
            parsed = _time.strptime(candidate, "%H:%M")
        except (TypeError, ValueError):
            logger.warning(
                "digest.times: невалидное время %r (ожидалось 'HH:MM') — "
                "слот пропущен",
                t,
            )
            continue
        norm = "%02d:%02d" % (parsed.tm_hour, parsed.tm_min)
        if norm != candidate:
            logger.warning(
                "digest.times: время %r нормализовано в %r", t, norm,
            )
        slots.append((parsed.tm_hour, parsed.tm_min))

    if not slots:
        logger.warning(
            "digest.times: ни одного валидного слота в %r — использую дефолт %s",
            raw, list(_DEFAULT_DIGEST_TIMES),
        )
        return default_slots
    return slots


def normalize_location(location: Optional[str]) -> str:
    """
    Нормализует location-строку для группировки (cap по локации).

    MAJOR 5-фикс: лёгкая канонизация БЕЗ словаря алиасов.

    Схема нормализации:
      1. strip() + lowercase.
      2. Пунктуация с пробелами вокруг → ", " (схлопывает "Moscow  Russia" в
         "Moscow, Russia"; убирает trailing punctuation "Moscow, Russia.").
      3. Trailing запятые/точки → убираются.
      4. Leading "the " (case-insensitive) → убирается.

    Принцип: LLM-инструкция в промпте задаёт формат; нормализация снимает
    вариации капитализации, лишние пробелы и trailing punctuation.
    Для ОДНОГО места LLM должен возвращать ОДНУ и ту же строку в пределах
    одного цикла — это контролируется промптом, а не словарём.

    Parameters
    ----------
    location : str or None
        Строка от LLM.

    Returns
    -------
    str
        Нормализованная строка (lowercase, без trailing punctuation/пробелов).
        Пустая входная строка → "" (вызывающий сам решает → 'other').
    """
    if not location:
        return ""
    s = location.strip()
    if not s:
        return ""
    # lowercase
    s = s.lower()
    # ПОРЯДОК ВАЖЕН: сначала схлопываем пробелы, потом punctuation.
    # Иначе "Moscow  Russia" (double-space, нет запятой) проходит step 1
    # без изменений и только схлопывает пробел → "Moscow Russia" (без запятой).
    # Сначала space collapse:
    s = re.sub(r"[ \t]+", " ", s)
    # Теперь punctuation с возможными пробелами вокруг → ", "
    # (схлопывает ";", ".", "," и варианты "x  ,  y" → "x, y")
    s = re.sub(r"\s*[.,;]\s*", ", ", s)
    # Trailing точки и запятые (с возможными пробелами перед ними)
    s = re.sub(r"[ \t]*[.][ \t]*$", "", s)
    s = re.sub(r"[ \t]*[,][ \t]*$", "", s)
    s = s.strip(" \t")
    # Leading "the " — артикль не несёт семантики
    s = re.sub(r"^(the)\s+", "", s, flags=re.IGNORECASE)
    return s


# ── Задача 2 (V2): эвристический префильтр перед LLM ─────────────────────────
# Бесплатное отсеивание заведомо нерелевантных items ДО батчинга LLM.
# Принцип: только явные, однозначные случаи. При сомнении — пропускаем в LLM.
# Urgent-маркеры (катастрофы, жертвы, атаки) — не отсеиваем НИКОГДА.

# ЖЁСТКИЕ маркеры: item ОДНОЗНАЧНО не по профилю → отсеивается с score=1.
# Берутся из CUT-списка BATCH_SYSTEM_PROMPT (llm/prompts.py).
# Категории: мнения/прогнозы/аналитика/слухи/"источники сообщают"/
#            светская хроника / суточные цены на топливо / релизы AI-моделей.
#
# ВАЖНО: используем r"" (raw strings), иначе "\bforecast\b" превращается
# в backspace+буквы и re.search не находит word boundary.
#
# Ревизии (Major-1/2):
#   analysis → только qualified-формы (не голое слово, ложные срабатывания на
#     "Toxicology analysis confirms poisoning…").
#   according to sources / источники сообщают — маркер НЕ достаточен сам по себе;
#     отсеивает только при отсутствии number/date/proper_noun (ужесточение гейта).
#   chatgpt → только в связке с release/launch/rollout/unveil (не голое упоминание).
#   .* в "источник.*сообща" / "star.*news" — ограничено .{0,40} для title
#     (жадный .* тянул весь title+summary → ложные срабатывания).
_PREFILTER_HARD_MARKERS = (
    # English: прогнозы, мнения, аналитика, слухи
    r"\bforecast\b", r"\bforecasts\b", r"\bopinion\b", r"\bopinions\b",
    # ANALYSIS — только qualified: expert/market/technical analysis (не голое слово,
    # которое ловит "Toxicology analysis confirms…" без фактов).
    r"\b(?:expert|market|technical)\s+analysis\b",
    r"\brumor\b", r"\brumors\b", r"\brumours?\b",
    # "sources say/said/report/claims"
    r"source[s]?\s+(?:sa[yis]|report[s]?|claim[s]?)",
    # "according to sources/insiders" — Major-2: маркер достаточен только при
    # наличии number/date/proper_noun (см. _prefilter_item).
    r"according\s+to\s+(?:sources?|insiders?)",
    # Russian: прогнозы, мнения, аналитика, слухи
    r"\bпрогноз\b", r"\bпрогнози\b", r"\bпрогнозов\b", r"\bпрогнозам\b",
    r"\bмнени[ея]\b", r"\bвзгляд\b", r"\bвзглядов\b",
    r"\bаналитик\b", r"\bаналитикой\b",
    r"\bслух\b", r"\bслухи\b", r"\bслухов\b",
    # "источники сообщают" — .{0,40} вместо .* (жадный захват на title+summary).
    r"\bисточник[а-я]*.{0,40}сообща",
    # Светская хроника — .{0,40} для ограничения контекста.
    r"\bcelebrity\b", r"\bcelebrities\b", r"\bgossip\b",
    r"\bhollywood\b.{0,40}news", r"\bstar\b.{0,40}news",
    r"\broyal\b.{0,40}family\b", r"\bweddings?\b",
    # Суточные цены на топливо (НЕ атаки на НПЗ, не дефицит — только рутинные цены)
    r"\bfuel\s+price\b", r"\bfuel\s+prices\b",
    r"\bgas\s+price\b", r"\bgas\s+prices\b",
    r"\bbrent\s+price\b", r"\bcrude\s+oil\s+price\b",
    r"\bprice\s+at\s+the\s+pump\b",
    r"\bбензин\b.*цена", r"\bцена\b.*бензин",
    r"\bтоплив\b.*цена", r"\bцена\b.*топлив",
    # Релизы AI-моделей (НЕ инциденты с AI) — только qualified-формы.
    # chatgpt — Major-2: убрано голое упоминание, только в связке с релизом.
    r"\bai\s+model.{0,20}(?:release|launch)",
    r"\bopenai\b.{0,20}(?:release|launch)",
    r"\bgpt-?\d.{0,20}(?:release|launch)",
    r"\bclaude\b.{0,20}(?:release|launch)",
    r"\bllm\b.{0,20}(?:release|launch)",
    r"\bchatgpt\b.{0,20}(?:release|launch|rollout|unveil)",
    r"\bmodel\s+launch\b",
)

# URGENT-маркеры: при наличии — item НЕ отсеивается префильтром,
# даже если есть hard-мркер. Катастрофы, жертвы, атаки — пропускаем в LLM.
_PREFILTER_URGENT_MARKERS = (
    r"\bcasualt", r"\bdead\b", r"\bkilled\b", r"\binjured\b", r"\bwounded\b",
    r"\bevacuation\b", r"\bevacuee",
    # Major-2: attack — смягчён (без обязательной привязки city/town/village/
    # civilian), но НЕ убран совсем — одиночный attack всё ещё блокирует отсеивание.
    r"\battack\b",
    r"\bairstrike\b", r"\bmissile\b.*\bstrike\b",
    r"\bexplosion\b", r"\bblast\b",
    r"\bfire\b", r"\bwildfire\b", r"\bflood\b", r"\bflooding\b",
    r"\bearthquake\b", r"\btremor\b",
    r"\bradiation\b", r"\bnuclear\b",
    r"\bcrash\b", r"\bcrashed\b",
    r"\bterrorist\b", r"\bshooting\b", r"\bhostage\b",
    r"\bdisaster\b", r"\bcatastroph",
    # Major-2: расширенный набор погодных/инфраструктурных инцидентов.
    r"\bhurricane\b", r"\bstorm\b",
    r"\bblackout\b", r"\boutage\b",
    r"\bchemical\b", r"\bhazmat\b",
    # CRITICAL-фикс: русские маркеры катастроф/ЧП (без \b на конце — падежные формы).
    r"\bвзрыв", r"\bпогиб", r"\bжертв", r"\bранен", r"\bпострада",
    r"\bпожар", r"\bобстрел", r"\bракет", r"\bдрон", r"\bбпла",
    r"\bэвакуац", r"\bнаводнен", r"\bураган", r"\bшторм", r"\bавари",
    r"\bкатастроф", r"\bрадиац", r"\bхимическ", r"\bутечк", r"\bтеракт",
    r"\bштурм", r"\bмобилизац",
    # CRITICAL: дефицит (реальная новость профиля — дефицит бензина/товаров).
    r"\bдефицит",
)


def _prefilter_item(item) -> bool:
    """
    Эвристический префильтр: заведомо не наше → score=1 без вызова LLM.

    Возвращает True если item отсеивается, False если идёт в LLM.

    Консервативный дизайн:
      — только явные, однозначные маркеры из CUT-списка;
      — эвристика "нет ни цифр, ни имён, ни мест, ни дат" — ТОЛЬКО
        в комбинации с ≥1 жёстким маркером;
      — urgent-маркеры блокируют отсеивание (катастрофы не теряем).

    Схема работы (pipeline.py):
      — Отсеянные: state='skipped', llm_score=1, llm_reason='prefilter'.
      — llm_score=1 < guarantee_min_score (default 3) → гарантия НЕ подхватит.
      — Без бюджетного вызова LLM — экономия токенов.

    Major-1: has_proper_noun считается по СЫРЫМ title/summary (до .lower()).
    Major-2: "according to sources" / "источники сообщают" — НЕ достаточен сам
    по себе; отсеивает только если контент не содержит number/date/proper_noun.
    """
    # Собираем текст для маркеров (case-insensitive).
    raw_title = getattr(item, 'title', None) or ""
    raw_summary = getattr(item, 'summary', None) or ""
    lower_text = (raw_title + " " + raw_summary).lower()

    # 1. Проверяем urgent — если есть, НЕ отсеиваем.
    for pat in _PREFILTER_URGENT_MARKERS:
        if re.search(pat, lower_text, re.IGNORECASE):
            return False

    # 2. Ищем жёсткие маркеры. Break сразу после первого матча.
    #    "according to sources" — слабый маркер: only отсеивает если есть контент.
    #    Всё остальное — жёсткий маркер (отсеивает даже без контента).
    hard_hit = False  # True = any strong marker matched
    weak_hit = False  # True = only "according" matched
    for pat in _PREFILTER_HARD_MARKERS:
        if re.search(pat, lower_text, re.IGNORECASE):
            if "according" in pat:
                weak_hit = True
                continue          # не рвём цикл — ищем сильные маркеры
            hard_hit = True
            break

    if not hard_hit and not weak_hit:
        return False

    # 3. Проверяем "контент": число, место, дата — по lower_text.
    has_number = re.search(r"\d", lower_text)
    # has_location: проверяем ТОЛЬКО title (не summary).
    # Это защищает от ложного срабатывания generic location-words в summary
    # (напр. "across regions" в fuel price report не спасает мусорный item).
    # Но конкретная локация в title ("explosion in Voronezh region") — реальный
    # контекст, пропускаем в LLM.
    lower_title = raw_title.lower()
    has_location = re.search(
        r"\b(?:city|town|country|countries|village|county|state|states|region|area)\b"
        # CRITICAL-фикс: русские маркеры — требуем кириллицу слева, чтобы
        # Latin "region"/"city" не давали ложное срабатывание через
        # Unicode \b-normalization (Cyrillic "о" = Latin "o").
        # Ищем слово без латиницы слева: (?<![a-zA-Z]).
        r"|(?:(?<![a-zA-Z])\u0433\u043e\u0440\u043e\u0434[а-я]*\b|(?<![a-zA-Z])\u043e\u0431\u043b\u0430\u0441\u0442[а-я]*\b|"
        r"(?<![a-zA-Z])\u0440\u0430\u0439\u043e\u043d[а-я]*\b|(?<![a-zA-Z])\u0440\u0435\u0433\u0438\u043e\u043d[а-я]*\b|"
        r"(?<![a-zA-Z])\u0441\u0435\u043b[оа-я]*\b|(?<![a-zA-Z])\u0434\u0435\u0440\u0435\u0432\u043d[а-я]*\b|"
        r"(?<![a-zA-Z])\u043a\u0440\u0430[аиейюя]\b|(?<![a-zA-Z])\u043d\u043f\u0437\b)",
        lower_title, re.IGNORECASE,
    )
    has_date = re.search(
        r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}"
        r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}"
        r"|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)",
        lower_text, re.IGNORECASE,
    )

    # Major-1 + Major-PN-fix: proper noun — по СЫРОМУ тексту (до .lower()),
    # чтобы заглавные буквы в "Russia", "Gazprom", "ChatGPT" не потерялись.
    #
    # Финальный алгоритм (Major-PN-fix):
    #   PN = заглавный токен (≥3 букв), непосредственно ПЕРЕД которым стоит
    #   строчный токен-слово. Сканируется title + summary.
    #   Капитализированное слово в начале предложения (после punct) тоже
    #   считается PN, т.к. перед ним стоит punct — а после punct может идти
    #   слово с любой регистром (в т.ч. глагол/союз).
    #
    #   Игнорируемые capitalized-стоп-слова: The, A, An, In, On, At, By, For,
    #   From, With, And, Or, But, After, Before, This, These, It, Its, As, If,
    #   While, When (в любом регистре).
    #
    # Примеры:
    #   "Sources say Sudan army advances" → "Sudan" после "say" (строчное) → PN. ✓
    #   "Expert Opinion: economic"       → "Opinion:" (stop-word не срабатывает,
    #                                       т.к. "Expert" — первое слово,
    #                                       "Opinion" — capitalized, но не после
    #                                       строчного слова) → no PN → отсеивается. ✓
    #   "Gazprom deal, Moscow"            → "Moscow" после "," → PN. ✓
    #   "ChatGPT outage"                  → "outage" не cap, "ChatGPT" — cap но
    #                                       перед ним нет строчного слова → no PN. ✓
    #   "many analysts believe Russia"   → "Russia" после "believe" (строчное) → PN. ✓
    #   "Breaking News: attack"          → "News" — cap, но перед ней "Breaking"
    #                                       (cap, не строчное), нет PN → отсеивается. ✓
    #   "The Ukraine situation"           → "Ukraine" после "The" (stop-word) → no PN. ✓
    #   Russian: "Источники сообщают о взрыве" → urgent-маркер блокирует. ✓
    #   "Источники сообщают о ситуации в регионе" → location → не отсеивается. ✓
    _PN_STOP = {
        "the", "a", "an", "in", "on", "at", "by", "for", "from", "with",
        "and", "or", "but", "after", "before", "this", "these", "it", "its",
        "as", "if", "while", "when",
        # CRITICAL: stop-words после punctuation в title: "Exclusive:", "Breaking:",
        # "Update:", "Alert:", "Report:" — prev_word в STOP, чтобы следующее
        # capitalized слово НЕ считалось PN в title-контексте.
        "exclusive", "breaking", "update", "alert", "flash",
        "report", "briefing", "summary",
        # "according to sources/insiders" — weak маркер. Все токены
        # этой конструкции должны быть в _PN_STOP, чтобы последующие PN
        # детектились по их собственному контексту, а не по "according".
        "according",
        # FIX 2 (conservative): русские generic-слова, которые могут
        # капитализироваться в заголовках, но не являются PN.
        # "Рубля" после "курс" — обычное существительное (genitive case),
        # NOT PN. Без списка → false positive → item проходит префильтр →
        # LLM-бюджет на мусор.
        "рубля", "рубль", "рубли", "доллара", "доллар", "евро",
        "банка", "банк", "рынка", "рынок", "цены", "цена",
        "страны", "страна", "города", "город", "компании", "компания",
        "мира", "мир", "власти", "власть", "лидера", "лидер",
        "плана", "план", "计划",  # китайский дубликат (если попадёт)
        "daily", "fuel", "price",
        # FIX 2 (conservative): русские стоп-слова — generic capitalized русские
        # существительные/прилагательные в начале фразы, не PN.
        # Без списка → capitalized-русское слово в начале фразы детектится как PN
        # в comma/punct-backtrack (для смешанных заголовков с латиницей).
        "экспертное", "эксперт", "мнение", "мнений",
        "источник", "источники", "сообщают", "говорят",
        "данные", "данных", "информация", "новости",
        # Generic English words in comma-backtrack: "sources" after comma
        # (e.g., "Gazprom, sources say...") should not be a PN.
        "sources",
    }
    combined_text = (raw_title or "") + " " + (raw_summary or "")

    def _is_proper_noun(word: str, after_sop: bool, appears_lower: bool) -> bool:
        """
        Определяет, является ли word собственным именем (PN).

        after_sop=True  — между word и началом title/summary НЕТ строчных слов
                          (word начинает предложение / фразу) → обычное существительное.
        after_sop=False — перед word было ≥1 строчное слово → это PN.

        appears_lower=True — слово также встречается в lower-форме в тексте
                          → обычное English-существительное (не PN).

        PN-сигналы (в порядке приоритета):
          1. ≥1 заглавная ПОСЛЕ первой буквы (Russia, Gazprom, ChatGPT, Sudan) → PN.
          2. All-caps 2–3 буквы (EU, US, NATO, G7) → PN.
          3. Capitalized-first (Gazprom, Moscow, Voronezh, Tech, Russia):
               NOT appears_lower + NOT common-english → PN.
               (в т.ч. после after_sop — структурные заголовки "Exclusive: Gazprom").
        """
        if any(c.isupper() for c in word[1:]):
            return True
        if len(word) >= 2 and word.isupper() and len(word) <= 3:
            return True
        # Capitalized-first only: Gazprom, Moscow, Voronezh, Tech, Russia
        if len(word) >= 3 and word[0].isupper() and word.lower() != word:
            if appears_lower:
                return False  # слово встречается в lower-форме — обычное существительное
            # Не common English → считаем PN (включая after_sop: заголовки вида "Exclusive: Gazprom")
            common_english = {
                "opinion", "outlook", "sources", "analysis", "economic",
                "expert", "market", "forecast", "report", "update", "news",
                "times", "today", "year", "week", "month", "million", "billion",
                "deal", "plan", "view", "response", "comment", "statement",
                "break", "exclusive", "announcement", "information",
                "according", "insiders",
                "celebrity", "celebrities", "gossip", "royal", "weddings",
                "wedding", "hollywood", "star",
                # generic English determiners/adjectives — capitalized в начале
                # предложения, но не PN
                "many", "few", "several", "some", "most", "all", "any",
                "both", "each", "every", "other", "such", "same",
                "next", "last", "first", "second", "third",
                "current", "former", "latter", "latest", "new",
                "local", "global", "national", "international", "regional",
                "major", "minor", "final", "initial", "additional",
                # CRITICAL-фикс: "fuel"/"gas"/"oil" — capitalized mid-sentence in
                # "Fuel prices fell" / "Gas prices rise" — NOT PN. prev_word="markets"
                # (lowercase, not stop) → prev_word="" → ложный PN. Добавлены в
                # common_english → not PN → мусорный item отсеивается. ✅
                "fuel", "gas", "oil", "crude",
                # stop-words после punctuation в title: "Exclusive:", "Breaking:",
                # "Update:", "Alert:" — следующее capitalized слово может быть PN.
                "exclusive", "breaking", "update", "alert", "flash",
                "report", "briefing", "summary",
            }
            if word.lower() not in common_english:
                return True
        return False

    def _find_proper_noun(text: str) -> bool:
        """
        Ищет PN: capitalized-слово(>=3 букв), перед которым стоит строчное слово.

        Prev-word anchored capitalization:
          1. Находим пару: строчное слово (curr) + capitalized слово (nxt).
          2. Пропускаем punct и stop-words перед curr, ищем prev_word.
          3. nxt = PN если prev_word из _PN_STOP или prev_word не найден
               ("Sources say Sudan", "According to sources, Gazprom").
               prev_word capitalized (не stop) -> NOT PN
               ("Exclusive: Gazprom", "Daily Fuel Price Report").
        """
        tokens = re.findall(r"[A-Za-zА-ЯЁа-яё]+|[^\s\w]", text)
        for idx in range(len(tokens) - 1):
            curr = tokens[idx]
            nxt = tokens[idx + 1]
            # Skip punctuation/non-alpha tokens: "Exclusive:" has ':' as curr.
            # Without this, ':' passes the STOP-check (':'.lower()=' ': not in STOP)
            # and Gazprom is tested as PN-candidate with prev_word=''.
            # Comma as curr: check if prev (alpha) token is a PN.
            # Handles "Gazprom, according to sources" — backtrack for comma
            # reaches "Gazprom" before the STOP-word "according" can block it.
            if curr == ',' and idx > 0:
                # Backtrack: skip punctuation AND non-PN non-STOP words until
                # we find a PN-candidate or a STOP word.
                # "Gazprom, nears, deal, according...":
                #   j=4: t='deal' not-STOP, not-PN → continue (don't break)
                #   j=3: t='nears' not-STOP, not-PN → continue
                #   j=2: t='Gazprom' not-STOP, IS PN → prev_word_comma='Gazprom', break
                #   → return True
                # Genre-prefix check: if first alpha after punctuation is in _PN_STOP → NOT PN.
                # Example: "Gazprom, nears, deal" → ',' at idx, backtrack finds "nears"
                # not in _PN_STOP → continue to PN-check.
                # (FIX B: "sources" и др. латинские стоп-слова УБРАНЫ из _PN_STOP.)
                _first_alpha_after_comma = ""
                for _j in range(idx - 1, -1, -1):
                    _t = tokens[_j]
                    if not _t.isalpha():
                        _pcount = 1
                        while _j > 0:
                            _j -= 1
                            _t = tokens[_j]
                            if _t.isalpha():
                                break
                            _pcount += 1
                            if _pcount > 2:
                                _t = ""
                                break
                    if _t.isalpha():
                        _first_alpha_after_comma = _t
                        break
                if _first_alpha_after_comma:
                    _stripped_first = _first_alpha_after_comma.lower().strip(":,!?.…—–-")
                    if _stripped_first in _PN_STOP:
                        # Genre prefix — capitalized STOP before PN → NOT PN.
                        continue
                prev_word_comma = ""
                for j in range(idx - 1, -1, -1):
                    t = tokens[j]
                    if not t.isalpha():
                        # Skip punctuation — max 2 consecutive punct
                        punct_count = 1
                        while j > 0:
                            j -= 1
                            t = tokens[j]
                            if t.isalpha():
                                break
                            punct_count += 1
                            if punct_count > 2:
                                t = ""
                                break
                        if not t or not t.isalpha():
                            break
                    t_stripped = t.lower().strip(":,!?.…—–-")
                    if t_stripped in _PN_STOP:
                        prev_word_comma = ""
                        break
                    # Non-STOP word: check if it's a PN candidate
                    if _is_proper_noun(t, False, False):
                        prev_word_comma = t
                        break
                    # Not STOP, not PN → skip it and continue searching
                if prev_word_comma:
                    return True
            # Handles "word (PN)" — opening paren is followed by a PN candidate.
            # Backtrack: check if the word before '(' is a PN.
            if curr == '(' and idx > 0:
                prev_word_comma = ""
                for j in range(idx - 1, -1, -1):
                    t = tokens[j]
                    if not t.isalpha():
                        punct_count = 1
                        while j > 0:
                            j -= 1
                            t = tokens[j]
                            if t.isalpha():
                                break
                            punct_count += 1
                            if punct_count > 2:
                                t = ""
                                break
                        if not t or not t.isalpha():
                            break
                    t_stripped = t.lower().strip(":,!?.…—–-")
                    if t_stripped in _PN_STOP:
                        prev_word_comma = ""
                        break
                    # Check PN first (same priority as comma-backtrack).
                    if _is_proper_noun(t, False, False):
                        prev_word_comma = t
                        break
                if prev_word_comma:
                    return True
            if not curr.isalpha():
                continue
            if curr.lower() in _PN_STOP:
                continue
            # curr — alpha, not in _PN_STOP. Check if nxt is PN candidate.
            if len(nxt) >= 3 and _is_proper_noun(nxt, False, False):
                # nxt — capitalized candidate. backtrack ищет предшествующее слово.
                # Правило: останавливаемся на первом alpha-токене.
                # STOP-words пропускаются (ищем дальше). non-PN alpha тоже останавливает
                # (это prev_word). PN-alpha пропускается (продолжаем).
                prev_word = ""
                for j in range(idx - 1, -1, -1):
                    t = tokens[j]
                    if not t.isalpha():
                        # Skip punctuation — "Gazprom, according" and "Gazprom: situation".
                        # Skipping ',' (depth=1) and ':' (depth=2) reaches "Gazprom".
                        # Limit: max 2 consecutive punct to avoid skipping too far.
                        punct_count = 1
                        while j > 0:
                            j -= 1
                            t = tokens[j]
                            if t.isalpha():
                                break
                            punct_count += 1
                            if punct_count > 2:
                                t = ""
                                break
                        if not t.isalpha():
                            break
                    t_stripped = t.lower().strip(":,!?.…—–-")
                    if t_stripped in _PN_STOP:
                        # STOP-word — пропускаем, ищем дальше.
                        continue
                    # First alpha-not-STOP word. Проверяем: это PN-токен?
                    if _is_proper_noun(t, False, False):
                        # PN-токен — пропускаем (считаем его "фоном"), ищем дальше.
                        continue
                    # non-PN alpha → это prev_word. Останавливаемся.
                    prev_word = t
                    break
                # curr alpha (lowercase/non-STOP) перед PN → prev_word найден.
                # Контракт (docstring :585-587):
                #   prev_word в _PN_STOP или prev_word="" → PN подтверждён.
                #   prev_word capitalized (не stop) → NOT PN ("Exclusive: Gazprom").
                if prev_word and prev_word[0].isupper():
                    stripped = prev_word.lower().strip(":,!?.…—–-")
                    if stripped not in _PN_STOP:
                        # capitalized non-STOP prev_word → NOT PN, continue searching.
                        continue
                return True
        return False

    # FIX 2: PN детектится по capitalized-after-lowercase паттерну.
    # Для чисто русских заголовков (без латиницы) _find_proper_noun сканирует
    # title на кириллические PN ("Газпром", "Россия") mid-sentence.
    # Ограничение: capitalized-слово в начале фразы ("Экспертное")
    # не считается PN (no lowercase before it), поэтому item отсекается.
    # PN-детект делается ТОЛЬКО по raw_title (НЕ combined_text):
    # summary-упоминания — фоновый контекст, не новостной факт.
    has_proper_noun = _find_proper_noun(raw_title)
    # При наличии любого {число, имя, место, дата} -- пропускаем в LLM.
    if has_number or has_proper_noun or has_location or has_date:
        return False

    # Отсеиваем: контент пустой (нет number/PN/location/date).
    # hard_hit -- отсеиваем (жёсткий маркер без фактов).
    # has_weak_only -- отсеиваем ("according to sources" без фактов).
    return True


# generic-суффиксы для нормализации location-имён в cap-группировке.
# "(2) Отбрасывание generic-суффиксов" в _normalize_cap_key.
_CAP_GENERIC_PATTERNS = (
    # NOTE: "area" intentionally excluded — "Bay Area" is a specific region,
    # not generic. Stripping "area" would collapse "Bavaria area" → "bavaria"
    # (incorrect; test_digest_guarantee_cap::test_generic_suffixes_stripped).
    r"\b(?:region|oblast|krai|rayon|district|province)\b",
    r"\bобл[а-я]*\b", r"\bкрай\b", r"\bрайон\b",
    r"\bобласть\b", r"\bрегион\b",
)


def _normalize_cap_key(location: str) -> str:
    """
    Нормализует location-строку для группировки в location-cap.

    F3-фикс: «Voronezh region, Russia» ≠ «Voronezh» в старом коде →
    cap по локации не работал, все 4 новости про Воронеж проходили
    при max_per_location=2.

    Схема:
      1. lowercase + strip.
      2. Обрезка по первой запятой → берём только «ядро» перед страной/деталями.
         «Voronezh, Russia» → «voronezh»; «Voronezh region, Russia» →
         «voronezh region» (запятая после region → ядро «voronezh region»).
      3. Отбрасывание generic-суффиксов (region, oblast, область, край, …).

    Риск ложного схлопывания: «Москва» vs «Московская область».
    «московская область» → comma-cut не срабатывает → «московская область» →
    суффикс «область» убирается → «московская». «москва» → «москва». Это
    разные ключи → НЕ схлопываются. ✅

    Применяется одинаково в _loc_day_key и _cap_by_location → инвариант
    группировки сохраняется.
    """
    if not location:
        return ""
    raw = location.strip()
    if not raw:
        return ""
    # (1) Обрезка по первой запятой — НЕ для кодов штатов/стран.
    # Обрезка нужна для «Voronezh region, Russia», но НЕ для «Portland, OR»
    # или «Portland, ME» (разные бакеты). Схема: режем только если после
    # запятой НЕ 2-3 заглавные буквы (US state / country code) и не
    # известное цельное название (несколько слов с мажорной буквой).
    # Проверяем РЕГИСТР на исходной строке ДО lower(): после lower() все
    # буквы строчные и [A-Z] не срабатывает → "Portland, OR" и
    # "Portland, ME" схлопывались в "portland".
    comma_pos = raw.find(",")
    if comma_pos >= 0:
        after_raw = raw[comma_pos + 1:].strip()
        # 2-3 заглавные буквы — код штата (OR, ME) или страны (US, UK, DE)
        is_state_or_country = bool(
            re.match(r"^[A-Z]{2,3}(,|$)", after_raw)
        )
        # Несколько заглавных слов — «United Arab Emirates», «New York»
        is_full_name = len(after_raw.split()) > 1 and all(
            w[0].isupper() for w in after_raw.split() if w
        )
        if not is_state_or_country and not is_full_name:
            raw = raw[:comma_pos].strip()
    s = raw.lower()
    if not s:
        return ""
    # (2) Отбрасывание generic-суффиксов
    for pat in _CAP_GENERIC_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    return s.strip()


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
    location = (parts[0] if parts else 'other').strip()
    # F3-фикс: нормализуем ядро локации для cap'а (схлопываем вариации).
    # Без этого «Voronezh region» ≠ «Voronezh» → cap пропускал все 4 новости.
    location = _normalize_cap_key(location) or 'other'
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

    F4-фикс: перед source-first проверкой добавляется content-override.
    Если title/summary/tags содержат политико-экономические маркеры
    (election, president, government, war, sanctions, …) и НЕТ профильных
    маркеров для данного источника (для cinema: film, movie, oscar, …),
    возвращается 'other' → секция «Основные новости».
    Это блокирует кейс Variety → «Lula's Election Win…» → political news
    попадала в 🎬 Кино вместо 📰 Основные новости.
    """
    raw = (item.source or "").lower()
    import re
    # Убираем всё что не буква/цифра (включая Unicode-апострофы, тире, пробелы)
    # "Tom's Hardware" → "tomshardware"; "Krebs On Security" → "krebsonsecurity"
    normalized = re.sub(r"[^\w]", "", raw, flags=re.UNICODE)

    source = _SOURCE_ALIASES.get(normalized, normalized)

    tags = _parse_tags(getattr(item, 'tags', None))
    tags_lower = [t.lower() for t in tags]

    # ── F4: собираем текст для content-check ────────────────────────────────
    # Для content-override собираем текст из title, summary, tags
    # (достаточно title + summary, tags обрабатываются отдельно).
    title_text = (getattr(item, 'title', None) or "").lower()
    summary_text = (getattr(item, 'summary', None) or "").lower()
    combined_lower = (title_text + " " + summary_text + " " + " ".join(tags_lower)).lower()

    # F4: политико-экономические маркеры → content_override = 'other'
    # (новость о реальных событиях, не entertainment-контент).
    _POLITICAL_MARKERS = (
        # English
        r"\belection\b", r"\belections\b", r"\bpresident\b", r"\bprime\s*minister\b",
        r"\bgovernment\b", r"\bparliament\b", r"\bcongress\b", r"\bsenate\b",
        r"\bminister\b", r"\bchancellor\b", r"\bgovernor\b",
        r"\bwar\b", r"\bconflict\b", r"\binvasion\b",
        r"\bsanction\b", r"\bsanctions\b", r"\btariff\b", r"\btariffs\b",
        r"\bpolicy\b", r"\bpolicies\b", r"\blegislation\b",
        r"\blawsuit\b", r"\bcourt\b", r"\btrial\b", r"\bverdict\b", r"\barrest\b",
        r"\bdiplomat\b", r"\bdiplomacy\b", r"\bnato\b", r"\beu\b",  # EU standalone
        r"\bg20\b", r"\bgaffairs\b", r"\bpolitics\b", r"\bpolitical\b",
        r"\bvote\b", r"\breferendum\b", r"\bcampaign\b",
        r"\bpeace\b", r"\bceasefire\b", r"\btruce\b",
        r"\boffensive\b", r"\bassault\b", r"\battack\b", r"\bairstrike\b",
        r"\bputin\b", r"\bzelensky\b", r"\bbiden\b", r"\btrump\b",
        # Russian
        r"\bвыбор[ыаеов]\b", r"\bпрезидент\b", r"\bправительств",
        r"\bпарламент\b", r"\bдепутат\b", r"\bминистр\b", r"\bгубернатор\b",
        r"\bвойн[аеуы]\b", r"\bконфликт\b", r"\bвоенн",
        r"\bсанкци[яеи]\b", r"\bтаможен", r"\bпошлин",
        r"\bзакон\b", r"\bзаконопроект\b", r"\bзаконодательств",
        r"\bсудебн", r"\bсуд\b", r"\bприговор\b", r"\bарест\b",
        r"\bдипломат\b", r"\bдипломати",
        r"\bнато\b", r"\bполитик\b", r"\bполитическ",
    )

    # F4: профильные маркеры для domain-specific источников.
    # Если они присутствуют — content НЕ переопределяется на 'other'.
    _CINEMA_MARKERS = (
        r"\bfilm\b", r"\bmovie\b", r"\bbox\s*office\b", r"\boscars?\b",
        r"\bcannes\b", r"\bstreaming\b", r"\bseries\b", r"\bseason\b",
        r"\bstudio\b", r"\bcinema\b", r"\bdirector\b", r"\bactor\b",
        r"\bactress\b", r"\bpremiere\b", r"\btrailer\b", r"\bnetflix\b",
        r"\bhollywood\b", r"\bemmy\b", r"\bbafta\b", r"\bgrammy\b",
        r"\bфильм\b", r"\bсериал\b", r"\bпремьер[аеу]\b", r"\bкинотеатр\b",
        r"\bстудия\b", r"\bрежисс[её]р\b", r"\bакт[ёе]р\b", r"\bсъёмк[аи]\b",
    )
    _GAMES_MARKERS = (
        r"\bgames?\b", r"\bgaming\b", r"\bplaystation\b", r"\bxbox\b",
        r"\bnintendo\b", r"\bsteam\b", r"\bepic\s*games\b", r"\bblizzard\b",
        r"\bue\b(?:\s*5|\s*4|\s*engine)", r"\bunreal\b",
        r"\bdlc\b", r"\bpatch\b", r"\bupdate\b", r"\brelease\s*date\b",
    )
    _SPACE_MARKERS = (
        r"\brocket\b", r"\bsatellite\b",
        r"\bastronaut\b", r"\bcosmonaut\b", r"\blaunch\b",
        r"\bspacex\b", r"\borbit\b", r"\bmission\b",
        r"\blunar\b", r"\bmars\b", r"\bstar\b",
        # Отдельные специфичные маркеры (без generic "space" который
        # слишком часто встречается в NASA-политических новостях:
        # "space program funding", "space agency cut" и т.п.)
    )

    # Определяем, есть ли политический контент
    has_political = any(
        re.search(pat, combined_lower) for pat in _POLITICAL_MARKERS
    )

    # Профильные маркеры — зависят от того, какую категорию source would дать
    has_cinema_profile = any(re.search(p, combined_lower) for p in _CINEMA_MARKERS)
    has_games_profile = any(re.search(p, combined_lower) for p in _GAMES_MARKERS)
    has_space_profile = any(re.search(p, combined_lower) for p in _SPACE_MARKERS)

    # F4: content-override для domain-specific источников с политическим контентом
    # «Variety: Lula's Election Win» → Variety source would give 'cinema',
    #   but has political=True + has_cinema_profile=False → return 'other'
    # «Variety: New Marvel Movie Trailer» → has political=False → source-first
    #   check applies → 'cinema' (unchanged behavior)
    _SOURCE_TO_PROFILE = {
        'cinema': has_cinema_profile,
        'games': has_games_profile,
        'space': has_space_profile,
        # tech, cyber, science — не domain-specific entertainment-источники,
        # content-override для них не нужен (редко путают контент с другими темами)
    }
    for cat_check, has_profile in _SOURCE_TO_PROFILE.items():
        if source in {
            'cinema': ('deadline', 'variety', 'hollywoodreporter', 'screendaily'),
            'games': ('ign', 'eurogamer', 'dtf', 'gamespot', 'kotaku'),
            'space': ('spacenews', 'nasa', 'nasaspaceflight'),
        }.get(cat_check, ()):
            if has_political and not has_profile:
                return 'other'

    # ── Без content-override — штатная логика ───────────────────────────

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


def _build_cluster_key_base(
    location: str, stem: Optional[str], published: Optional[str],
    title: Optional[str] = None,
) -> str:
    """
    Внутренняя реализация сборки cluster_key.
    Публичная обёртка build_cluster_key экспортируется для совместимости
    (ранее — locations.build_cluster_key).
    """
    pub_day = published[:10] if (published and len(published) >= 10) else "unknown"
    if stem:
        return f"{location}|{stem}|{pub_day}"
    else:
        if title:
            title_hash = hashlib.md5(title.encode()).hexdigest()[:12]
        else:
            title_hash = hashlib.md5(b"").hexdigest()[:12]
        return f"{location}|nostem:{title_hash}|{pub_day}"


# Публичная обёртка — заменяет locations.build_cluster_key.
# Используется в тестах и для обратной совместимости.
def build_cluster_key(
    location: str, stem: Optional[str], published: Optional[str],
    title: Optional[str] = None,
) -> str:
    """
    Собирает cluster_key из компонентов.

    Parameters
    ----------
    location : str
        Каноническое название места (English). Нормализуется: lowercase + strip.
    stem : str or None
        Стем-хеш из title. None → nostem по title.
    published : str or None
        Дата в любом формате (используется первые 10 символов как YYYY-MM-DD).
    title : str, optional
        Исходный title для nostem-per-title hash.

    Returns
    -------
    str
        cluster_key: ``location|stem|day`` или ``location|nostem:hash|day``
    """
    # Нормализация: location всегда lowercase+strip для единообразия
    loc = location.strip().lower() if location else ""
    return _build_cluster_key_base(loc, stem, published, title)


def rebuild_cluster_key(
    llm_location: str, old_cluster_key: Optional[str],
    title: Optional[str] = None,
) -> str:
    """
    Пересобирает cluster_key с LLM-локацией вместо словарной.

    Первые два аргумента (локация и stem) теперь разнесены по времени:
    локация приходит от LLM ПОСЛЕ скоринга, stem вычислен при ингесте
    и закодирован в old_cluster_key.

    Для items без старого cluster_key возвращается ключ с LLM-локацией
    и nostem-per-title hash (все items без stem, но с разными title →
    разные keys → нет заражения score-propagation между новостями без stem).
    Для items с пустой LLM-локацией → 'other' (cap не применяется).

    Parameters
    ----------
    llm_location : str
        Каноничное название места от LLM (English, max-specific).
        Пустая строка = глобальная/непонятная тема.
    old_cluster_key : str or None
        Cluster_key, вычисленный при ингесте. Части stem и pub_day
        используются повторно.
    title : str, optional
        Исходный title item для nostem-per-title hash.
        Fallback на url обрабатывается вызывающей стороной (_process_batch)
        до вызова этой функции: ``title=getattr(item,'title',None) or
        getattr(item,'url',None)``. Если title всё ещё None → md5(b"")
        (крайний случай, очень редко на практике).

    Returns
    -------
    str
        Новый cluster_key: ``normalized_location|stem_or_nostem|pub_day``

    M1-фикс: nostem-per-title hash.
      Без title все nostem-items получали md5(b"") → ОДИНАКОВЫЙ cluster_key →
      один cluster → один score-propagation cycle → заражённый score.
      Теперь: md5(title_or_url)[:12] → разные title = разные keys = разные
      clusters → каждый item получает свой score-propagation slot.

    MAJOR 5-фикс: нормализация локации.
      (a) Промпт LLM запрашивает каноничный формат (english, «City, Country»);
          разные написания «Moscow», «MOSCOW», «Moscow, Russia», «Россия, Москва»
          должны давать одну строку.
      (b) Лёгкая нормализация: strip → lowercase → схлопывание множественных
          пробелов/запятых/точек. Без словаря алиасов — LLM-инструкция
          в промпте задаёт формат.
      (c) Ключ = нормализованная строка (полностью). Одинаковое место →
          одинаковый ключ → cap работает.
    """
    # (b) Лёгкая нормализация: normalize_location — strip/lower/collapse spaces/punct.
    #   «Moscow, Russia» → «moscow, russia»
    #   «Moscow  Russia» → «moscow, russia»  (double-space collapsed, comma added)
    #   «moscow , russia» → «moscow russia»  (trailing comma stripped)
    #   «Moscow, Russia.» → «moscow, russia» (trailing dot stripped)
    #   «The Moscow, Russia» → «moscow, russia» (leading the stripped)
    loc_norm = normalize_location(llm_location)

    if not loc_norm:
        loc_norm = "other"

    if old_cluster_key:
        parts = old_cluster_key.split("|")
        stem = parts[1] if len(parts) > 1 else ""
        pub_day = parts[2] if len(parts) > 2 else "unknown"
    else:
        stem = ""
        pub_day = "unknown"

    if stem:
        return f"{loc_norm}|{stem}|{pub_day}"
    else:
        # M1: nostem-per-title hash. title_or_url гарантирует уникальность
        # (разные title → разные hash → разные cluster_key → разные clusters).
        title_or_url = title if title else ""
        if not title_or_url:
            # Крайний fallback: md5 пустой строки даёт детерминированный,
            # но неуникальный ключ. Один на все items без title и без url.
            # Это приемлемый краевой случай (очень редкий на практике).
            title_hash = hashlib.md5(b"").hexdigest()[:12]
        else:
            title_hash = hashlib.md5(title_or_url.encode()).hexdigest()[:12]
        return f"{loc_norm}|nostem:{title_hash}|{pub_day}"


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
    # F3: наблюдаемость — отслеживаем коллизии нормализации.
    normalized_location_sources: Dict[str, List[str]] = {}

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
        location = (parts[0] if parts else 'other').strip()
        # F3-фикс: нормализуем ядро для ключа cap'а.
        location_norm = _normalize_cap_key(location) or 'other'
        pub_day = parts[2] if len(parts) > 2 else 'unknown'

        # F3: наблюдаемость — собираем маппинг нормализованный→оригинальные.
        if location_norm != 'other':
            normalized_location_sources.setdefault(location_norm, []).append(location)

        # bucket 'other' is not capped
        if location_norm == 'other':
            result.append(item)
            continue

        key = (location_norm, pub_day)
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

    # F3: наблюдаемость — warning если несколько исходных локаций
    # схлопнулись в одно нормализованное ядро (наблюдаемость, не меняет поведение).
    if len(normalized_location_sources) > 1:
        for key_norm, orig_locations in normalized_location_sources.items():
            if len(set(orig_locations)) > 1:
                uniq = sorted(set(orig_locations))
                logger.warning(
                    "Location cap: нормализованное ядро %r объединяет "
                    "%d вариантов локации: %s",
                    key_norm, len(uniq), ", ".join(uniq),
                )

    return result

def _category_guarantee(
    items: List[NewsItem],
    category_min: Dict[str, int],
    pool: Optional[List[NewsItem]] = None,
    new_fallback: Optional[List[NewsItem]] = None,
) -> Tuple[List[NewsItem], Set[str]]:
    """
    Гарантирует минимум новостей по категориям.

    Returns:
        Tuple[result_items, guarantee_added_urls]:
        - result_items: модифицированный список items
        - guarantee_added_urls: URLs items, добавленных из pool или new_fallback
          (все, кто закрывает минимум; не включает исходные items или evict'нутые)

    Алгоритм:
    1. Считаем текущее покрытие по категориям в items.
    2. Для каждой категории в category_min (кино первый): если покрытие < min,
       вытесняем лучшие по score items категории из pool,
       заменяя ими наименее ценные EVICTABLE items текущего набора.
       EVICTABLE = item, чья категория НЕ в category_min
       ИЛИ cat_counts[c] > category_min[c] (покрыта с запасом).
    3. После каждого swap инкрементально обновляем cat_counts.
    4. Fallback (new_fallback): если после шага 2–3 категория всё ещё ниже min —
       берём недостающее из нескоренных (state='new', llm_score=NULL).
       Это последний резерв; скоренные всегда приоритетнее.
    5. При пустом pool и None new_fallback — legacy-режим: ничего не меняется
       (swap невозможен, гарантия остаётся no-op).

    category_min нормализуется в Dict[str, int] (_normalize_category_min):
    строковые значения из YAML не должны давать TypeError в сравнении
    (Minor 1-фикс, wave 2).

    new_fallback сортирован fetched_at DESC (свежие первыми) вызывающей
    стороной (storage.get_new_items_for_fallback).

    Сорт вывода — по llm_score DESC вызывающей стороной после вызова.
    """
    category_min = _normalize_category_min(category_min)
    if not category_min or not items:
        return items, set()

    result = list(items)
    used_ids: Set[int] = {id(it) for it in items}
    guarantee_added_urls: Set[str] = set()
    original_ids: Set[int] = {id(it) for it in items}

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
            if id(candidate) not in original_ids:
                guarantee_added_urls.add(getattr(candidate, 'url', None) or id(candidate))
            added += 1

    # Шаг 4: new_fallback для оставшихся дефицитов.
    # new_fallback уже отсортирован fetched_at DESC (свежие первыми).
    if new_fallback:
        for cat, min_count in category_min.items():
            current = cat_counts.get(cat, 0)
            if current >= min_count:
                continue
            needed = min_count - current
            added = 0
            for candidate in new_fallback:
                if added >= needed:
                    break
                if id(candidate) in used_ids:
                    continue
                if _infer_category(candidate) != cat:
                    continue
                result.append(candidate)
                used_ids.add(id(candidate))
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                guarantee_added_urls.add(getattr(candidate, 'url', None) or id(candidate))
                added += 1

    # Финальный warning по невыполненным категориям
    for cat, min_count in category_min.items():
        current = cat_counts.get(cat, 0)
        if current < min_count:
            logger.warning(
                "Категория %s: минимум не достигнут (current=%d, min=%d)",
                cat, current, min_count,
            )

    return result, guarantee_added_urls


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
        self._max_per_cycle = max(0, _coerce_int(
            config.get("llm", {}).get("max_per_cycle", 150),
            150, "llm.max_per_cycle",
        ))
        self._llm_calls_left: int = self._max_per_cycle  # общий бюджет на цикл
        # off-peak гейт: ограничить LLM вызовами в непиковые часы DeepSeek
        # явный парсер: bool("false")==True, поэтому проверяем строковые значения
        off_peak_raw = config.get("llm", {}).get("off_peak_only", False)
        self._off_peak_only = _parse_bool(off_peak_raw)
        # лимит retry-попыток
        self._max_retries = _coerce_int(
            config.get("database", {}).get("max_retries", Storage.DEFAULT_MAX_RETRIES),
            Storage.DEFAULT_MAX_RETRIES, "database.max_retries",
        )
        # лимит stuck-pending для get_stuck_pending (защита RAM от большого backlog)
        self._stuck_pending_limit = _coerce_int(
            config.get("database", {}).get("stuck_pending_limit", 500),
            500, "database.stuck_pending_limit",
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
        retention = self.config.get("database", {}).get("retention_days", 1)
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
        window_days = _coerce_int(dedup_cfg.get("window_days", 3), 3, "dedup.window_days")
        recent_limit = _coerce_int(dedup_cfg.get("llm_recent_limit", 20), 20, "dedup.llm_recent_limit")

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

        # Critical-2-фикс: ingest safety — если модель пометила ВСЕ кандидаты
        # дублями при >1 кандидате, это ненормальный ответ (галлюцинация/потеря
        # батча). Без этой проверки все items получают state='skipped' и
        # пропадают безвозвратно.
        # Исключение: ровно 1 кандидат + все помечен — валидный cross-cycle
        # dedup одиночной новости (recent_list непуст в штатном сценарии).
        # Лечение: возвращаем бюджет, set_state не вызываем, items остаются new.
        if len(dup_0based) >= len(candidates) and len(candidates) > 1:
            self._llm_calls_left += num_chunks
            logger.warning(
                "LLM-дедуп (ingest): модель пометила все %d кандидатов дублями "
                "(>1) — игнорируем результат (бюджет %d возвращён). "
                "Items остаются state='new' для следующего цикла.",
                len(candidates), num_chunks,
            )
            return 0

        # Собираем URLs дублей
        dup_urls = [candidates[idx].url for idx in sorted(dup_0based) if idx < len(candidates)]

        # Critical #2-фикс: устанавливаем state='skipped' РЕАЛЬНОЙ перезаписью.
        # Для дублей, которые УЖЕ в БД со state='new' (через save_many):
        #   save_skipped = INSERT OR IGNORE → rowcount=0 → state остаётся 'new'
        #   → K1-recovery подхватит их в след.цикле или они застрянут навсегда.
        # set_state — UPDATE, который РЕАЛЬНО меняет state='new' → 'skipped'.
        # Minor #3-фикс: если UPDATE partially failed (DB locked и т.п.),
        # marked < len(dup_urls) — часть URLs осталась 'new' в БД.
        # На следующем цикле recovered-items не пройдут LLM-дедуп
        # (set обновился) → дубль пушится навсегда. Лечение: при расхождении
        # warning, исключаем из _llm_dedup_skipped_urls настолько, насколько
        # можем подтвердить (marked штук). Это уменьшает защиту, но не блокирует
        # её полностью.
        if dup_urls:
            marked = self.storage.set_state(dup_urls, "skipped")
            if marked < len(dup_urls):
                logger.warning(
                    "LLM-дедуп: set_state('skipped') обновил %d из %d строк — "
                    "%d URLs исключены из skipped-списка (могут повториться в след.цикле). "
                    "Причина: DB locked или несуществующие URLs.",
                    marked, len(dup_urls), len(dup_urls) - marked,
                )
                # Подтверждённые URLs (те что обновились) — upper bound:
                # реально обновлено ≤ marked ≤ len(dup_urls).
                confirmed = dup_urls[:marked] if marked > 0 else []
                self._llm_dedup_skipped_urls.update(confirmed)
            else:
                self._llm_dedup_skipped_urls.update(dup_urls)
                logger.debug(
                    "LLM-дедуп: set_state('skipped') для %d дублей → %d строк обновлено",
                    len(dup_urls), marked,
                )

        logger.info(
            "LLM-дедуп: отсеяно %d/%d кандидатов (семантические дубли «одно событие»), "
            "бюджет списан: %d (дедуп) + далее скоринг",
            len(dup_urls), len(candidates), num_chunks,
        )
        return len(dup_urls)

    def _check_against_shown(
        self,
        candidates: List[NewsItem],
        recent_shown: List[dict],
    ) -> Set[str]:
        """
        Проверяет пул кандидатов (backfill/guarantee/skipped) против показанных
        за окно. Один LLM-вызов detect_duplicates(candidates, recent_shown)
        на весь пул → O(1) LLM-вызовов вместо O(N).

        Returns: set of duplicate URLs.
        """
        if not candidates or not recent_shown:
            return set()
        llm_filter = getattr(self, "llm_filter", None)
        if llm_filter is None:
            return set()
        # MAJOR-фикс (Major #2): Cost-инвариант — в пик DeepSeek LLM не вызываем.
        # Паттерн идентичен _cluster_deferred_duplicates (pipeline.py:1133-1138).
        # Покрывает оба пути: запланированный дайджест и /digest из чата.
        off_peak_only = getattr(self, "_off_peak_only", False)
        if off_peak_only and not is_llm_allowed_now(off_peak_only):
            logger.info(
                "Дайджест: RECENT-дедуп пула пропущен (пик DeepSeek, off_peak_only)",
            )
            return set()
        # MAJOR-фикс (4c): один батчевый вызов на весь пул (backfill, skipped).
        # detect_duplicates снят len<2 guard при непустом RECENT.
        try:
            dup_idx = llm_filter.detect_duplicates(candidates, recent_shown)
        except Exception as e:
            logger.warning(
                "Дайджест: RECENT-дедуп пула (%d кандидатов) упал: %s — "
                "пропускаем (fail-open)",
                len(candidates), e,
            )
            return set()
        dup_urls: Set[str] = set()
        # При непустом RECENT лидер (i=0) может быть дублем показанного.
        for i in dup_idx:
            if isinstance(i, int) and 0 <= i < len(candidates):
                dup_urls.add(candidates[i].url)
        if dup_urls:
            logger.info(
                "Дайджест: backfill/guarantee RECENT-дедуп отсеял %d из %d",
                len(dup_urls), len(candidates),
            )
        return dup_urls

    def _cluster_deferred_duplicates(
        self, items: List[NewsItem], window_start_iso: str,
    ) -> Tuple[Set[str], List[dict]]:
        """
        Семантическая кластеризация deferred-новостей ПЕРЕД дайджестом.

        Финальный сейф от дублей: новости одного события, накопленные
        за несколько циклов сбора, группируются LLM; дублями считаются
        все, кроме лидера — первого в текущем порядке выдачи
        (наиболее приоритетного). Лидеры остаются в дайджесте.

        Кросс-оконный дедуп: для RECENT-контекста берутся новости,
        УЖЕ показанные в текущем дайджест-окне (state IN ('digested','delivered'),
        fetched_at >= window_start_iso). Это отсекает перефразированные дубли
        из предыдущего слота того же окна (утро → вечер). Кандидаты
        (``deferred``) НЕ попадают в RECENT автоматически (другой state).

        Returns
        -------
        Tuple[Set[str], List[dict]]
            (set of duplicate URLs, list of recent_shown dicts).
            recent_shown возвращается для повторного использования в
            backfill/guarantee (Fix 4c: всё в отправленном дайджесте
            прошло RECENT-дедуп).

        При любой ошибке / отсутствии LLM / выключенном флаге / пике DeepSeek —
        возвращает (empty set, empty list) — безопасно.

        Args:
            items: deferred-кандидаты текущего дайджеста.
            window_start_iso: граница текущего окна (ISO-8601, UTC).
                Все digested/delivered items с fetched_at >= window_start_iso
                попадают в RECENT-контекст для LLM-дедупа.
        """
        if not items:
            return (set(), [])

        # Кросс-оконный дедуп: собираем показанные за окно (digested + delivered).
        # Fail-open: ошибка storage → пустой RECENT (не рвём дайджест).
        # Fetched early so it's available for all early-return paths.
        try:
            recent_shown: List[dict] = self.storage.get_shown_for_window(
                cutoff_iso=window_start_iso,
                limit=200,
            )
        except Exception as e:
            logger.warning(
                "Дайджест: get_shown_for_window упала (cutoff=%s): %s — "
                "используем пустой RECENT-контекст",
                window_start_iso, e,
            )
            recent_shown = []

        # MAJOR-фикс (4a): с RECENT непустым — вызываем detect_duplicates
        # для одиночного кандидата (может быть дублем показанного).
        # Без RECENT и <2 кандидатов — early-return (лидеру не с чем быть дублем).
        if len(items) < 2 and not recent_shown:
            return (set(), recent_shown)

        digest_cfg = self.config.get("digest", {}) or {}
        if not _parse_bool(digest_cfg.get("cluster_duplicates", True)):
            return (set(), recent_shown)
        # Cost-инвариант: в пик DeepSeek LLM не вызываем (как _run_llm_dedup).
        off_peak_only = getattr(self, "_off_peak_only", False)
        if off_peak_only and not is_llm_allowed_now(off_peak_only):
            logger.info(
                "Дайджест: кластеризация дублей пропущена (пик DeepSeek, off_peak_only)",
            )
            return (set(), recent_shown)
        llm_filter = getattr(self, "llm_filter", None)
        if llm_filter is None:
            return (set(), recent_shown)

        try:
            dup_idx = llm_filter.detect_duplicates(list(items), recent_shown)
        except Exception as e:
            logger.warning("Дайджест: кластеризация дублей упала: %s", e)
            return (set(), recent_shown)
        if not isinstance(dup_idx, (set, frozenset)):
            return (set(), recent_shown)
        urls: Set[str] = set()
        for i in dup_idx:
            # M2-фикс: лидер (i=0) может быть помечен как дубль более нового
            # item'а из того же чанка — правило "newer becomes leader" (:537-539).
            # Инвариант «хотя бы один item остаётся» при пустом RECENT
            # обеспечивается ниже (:1365). Снимаем защиту i==0.
            if isinstance(i, int) and 0 <= i < len(items):
                urls.add(items[i].url)
        # Инвариант: хотя бы один item (лидер) должен остаться в дайджесте.
        # Действует ТОЛЬКО при пустом RECENT. Когда RECENT непуст,
        # ЛЕГИТИМНЫЙ сценарий «все кандидаты — пересказы утренних показанных»
        # → все {i} дублі → len(urls)==len(items), и отправлять НЕ нужно.
        # Без этого условия код «спасал» бы все дубли и отправлял их.
        if not recent_shown and len(urls) >= len(items):
            logger.warning(
                "Дайджест: кластеризация пометила все %d items дублями — "
                "игнорируем результат (лидер должен остаться)", len(items),
            )
            return (set(), recent_shown)
        if urls:
            logger.info(
                "Дайджест: кластеризация отсеяла %d дублей из %d кандидатов",
                len(urls), len(items),
            )
        else:
            # F: наблюдаемость — сообщаем что кластеризация прошла, но не нашла дублей.
            logger.info(
                "Дайджест: кластеризация проверила %d кандидатов — дубли не найдены",
                len(items),
            )
        return (urls, recent_shown)

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

        # Push-cap: счётчик на весь цикл (суммарно по всем _process_batch).
        # Сбрасывается в начале каждого цикла. Список [count] — shared ref
        # через все вызовы _process_batch → _apply_result.
        self._push_count_ref: List[int] = [0]

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
            # FIX-3: вызываем count_by_state вместо get_stuck_pending() без limit
            # (get_stuck_pending без limit грузит ВСЕ зависшие pending в RAM).
            pending_count = self.storage.count_by_state("pending")
            total_waiting = total_new_items_count + pending_count
            logger.info(
                "PAUSED gate: обработка пропущена, %d items ждут возобновления "
                "(new=%d, pending=%d)",
                total_waiting, total_new_items_count, pending_count,
            )
            # очистка ВЫПОЛНЯЕТСЯ (purge не зависит от LLM/доставки)
            self._cleanup()
            return  # ← выходим: без LLM, доставки, дайджеста, алертов

        if self._off_peak_only and not is_llm_allowed_now(self._off_peak_only):
            # вычисляем реальное число ожидающих items до return
            # FIX-3: вызываем count_by_state вместо get_stuck_pending() без limit
            # (get_stuck_pending без limit грузит ВСЕ зависшие pending в RAM).
            pending_count = self.storage.count_by_state("pending")
            total_waiting = self.storage.count_by_state("new") + pending_count
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

        # ── V2 prefilter (Major-3): ДО LLM-дедупа — экономия токенов на мусоре ──
        # llm.prefilter.enabled (default True). При false — пропускаем всех в LLM.
        # Отсеянные: state='skipped', llm_score=1, llm_reason='prefilter'.
        # llm_score=1 < guarantee_min_score (default 3) → гарантия не подхватит.
        # Применяется к all_items (recovered + fresh candidates) — fresh items из БД
        # (state='new', подхваченные в recovery K1) тоже проходят префильтр.
        prefilter_enabled = _parse_bool(
            self.config.get("llm", {}).get("prefilter", {}).get("enabled", True)
        )
        prefiltered_urls: List[str] = []
        if prefilter_enabled:
            for item in all_items:
                # Scored items (recovered/retry) уже имеют оценку — не трогаем.
                if getattr(item, 'llm_score', None) is None and _prefilter_item(item):
                    prefiltered_urls.append(item.url)
            if prefiltered_urls:
                # Фиксируем state + score/reason сразу — без вызова LLM.
                # Minor #3: используем return value (per-record изоляция).
                updated_urls = self.storage.update_llm_results_batch([{
                    "url": url,
                    "score": 1,
                    "summary": "",
                    "reason": "prefilter",
                    "urgent": False,
                    "cluster_key": None,
                    "title_en": None,
                } for url in prefiltered_urls])
                self.storage.set_state(updated_urls, "skipped")
                # Major-3: исключаем из all_items → дедуп не увидит мусор.
                prefilter_set = set(prefiltered_urls)
                all_items = [i for i in all_items if i.url not in prefilter_set]
                logger.info(
                    "Prefilter: %d items отсеяны до LLM-дедупа "
                    "(токены сохранены), %d → dedup",
                    len(prefiltered_urls), len(all_items),
                )

        # ── LLM-дедупликация: семантическая группировка «одно событие» ──
        # Critical #1-фикс: вызов ПОСЛЕ гейтов паузы и off-peak.
        # На паузе и в off-peak LLM-дедуп НЕ вызывается — экономия платных токенов.
        # Кандидаты уже сохранены в БД (state='new'), лексический дедуп и seen_titles
        # также уже выполнены (до гейтов) — сохранность данных обеспечена.
        # При сбое LLM — безопасный пропуск: ни один дубль не проходит.
        # Major-3: prefiltered URLs уже исключены из all_items.
        llm_dedup_skipped = 0
        # Проверяем что llm_filter имеет необходимые атрибуты (для тестов с FakeLLMFilter).
        has_dedup_chunk = (
            self.llm_filter is not None
            and hasattr(self.llm_filter, '_DEDUP_CHUNK_SIZE')
        )
        if self.llm_dedup_enabled() and has_dedup_chunk:
            llm_dedup_skipped = self._run_llm_dedup(all_items)
            # Исключаем LLM-дубли из дальнейшей обработки
            if llm_dedup_skipped > 0:
                dedup_skipped_urls = getattr(self, '_llm_dedup_skipped_urls', set())
                all_items = [i for i in all_items if i.url not in dedup_skipped_urls]
            if getattr(self, '_llm_dedup_skipped_urls', None):
                # Очищаем после использования
                self._llm_dedup_skipped_urls.clear()

        # ── B13: retry-очередь обрабатывается ПЕРЕД кандидатами ──
        # Major-3: candidates уже очищен от prefiltered и dedup-skipped URLs.
        # M1-фикс: scored items (llm_score уже сохранён) НЕ тратят LLM-бюджет.
        #   — Разделяем stuck на scored (retry без LLM) и fresh (требуют оценки).
        #   — Scored обрабатываются БЕЗ ограничения бюджетом.
        #   — Fresh обрабатываются в рамках retry_budget (как и раньше).
        # Это решает проблему «голодовки» при _llm_calls_left==0:
        #   при исчерпанном бюджете scored-pending всё равно доставляются.
        stuck = self.storage.get_stuck_pending(
            max_age_seconds=threshold,
            max_retries=self._max_retries,
            limit=getattr(self, "_stuck_pending_limit", 500),
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
                    stuck_scored, budget_override=len(stuck_scored),
                    push_count_ref=self._push_count_ref,
                )
                if failed_retry_scored:
                    self.storage.increment_retry_count(failed_retry_scored)
                    self._check_and_mark_failed(failed_retry_scored)
            else:
                failed_retry_scored = []

            # Fresh retry: бюджет по-старому (минимум половины или len, кламп)
            retry_budget: int
            if stuck_fresh:
                # B13-фикс: резервируется ≥½ бюджета, но свежие кандидаты
                # никогда не получают 0 при left>0 (даже если fresh_items > left//2).
                # FIX-4: max(..., 1) применяется только если left > 0 (иначе 0).
                if self._llm_calls_left > 0:
                    retry_budget = min(max(self._llm_calls_left // 2, 1), len(stuck_fresh))
                else:
                    retry_budget = 0
                if stuck_scored:
                    logger.info(
                        "B13: retry fresh %d items — бюджет %d/%d",
                        len(stuck_fresh), retry_budget, self._llm_calls_left,
                    )
                failed_retry_fresh = self._process_batch(
                    stuck_fresh, budget_override=retry_budget,
                    push_count_ref=self._push_count_ref,
                )
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
                all_items, budget_override=remaining_budget,
                push_count_ref=self._push_count_ref,
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
        push_count_ref: Optional[List[int]] = None,
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
            push_count_ref: shared cycle-level push counter [count] (Mutable).
                            If max_pushes_per_cycle > 0 and count >= limit → overflow → deferred.

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
        skipped_urls: Set[str] = set()
        deferred_urls: Set[str] = set()
        to_deliver: List[NewsItem] = []
        llm_batch_records: List[dict] = []

        if self.llm_filter is not None:
            push_th = self.llm_filter.push_threshold
            digest_th = self.llm_filter.digest_threshold
            push_urgent_only = self.llm_filter.push_urgent_only
            max_pushes_per_cycle: int = _coerce_int(
                self.config.get("llm", {}).get("max_pushes_per_cycle"), 0,
                "llm.max_pushes_per_cycle",
            )
            # Major-3: prefilter выполняется в run_cycle_with_delivery ДО _run_llm_dedup.
            # Здесь — только маршрутизация и сохранение результатов.

            # лимит items: бюджет передан явно (retry) или остаток общего счётчика
            budget = budget_override if budget_override is not None else self._llm_calls_left
            # Minor #4-фикс: при budget<0 → клампим в 0 (items[:<0] даёт непустой префикс)
            budget = max(0, budget)
            items_to_llm = items
            if budget <= 0:
                # бюджет исчерпан — все items возвращаются в state='new' (K1 подхватит
                # в следующем цикле). C1': НЕ добавляем в failed_urls →
                # retry_count не инкрементируется.
                logger.warning(
                    "LLM-бюджет цикла исчерпан (0/%d); %d items → 'new' (retry в след.цикле)",
                    self._max_per_cycle, len(items),
                )
                to_deliver = []  # никто не доставлен, никого не помечаем failed

            if len(items) > budget:
                logger.warning(
                    "LLM-бюджет %d/%d: обрабатываем первые %d items, "
                    "остальные %d — back to 'new' (retry в след.цикле)",
                    budget, self._max_per_cycle, budget, len(items) - budget,
                )
                items_to_llm = items[:budget]
                excess_urls = [i.url for i in items[budget:]]
                if excess_urls:
                    self.storage.set_state(excess_urls, "new")
                    logger.debug(
                        "Бюджетный срез: %d items возвращены в state='new' "
                        "(retry_count сохранён)",
                        len(excess_urls),
                    )

            # batch_size из конфига; 1 = поштучный режим (всегда evaluate)
            batch_size = max(1, _coerce_int(
                self.config.get("llm", {}).get("batch_size", 10),
                10, "llm.batch_size",
            ))

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
                        # M1: сохраняем llm_title из БД — retry НЕ должен его затирать.
                        # Без этого: FilterResult.title_en=None → llm_batch_records title_en=None
                        # → update_llm_results_batch стирал llm_title в NULL
                        # → следующий рендер fallback на RU-title.
                        title_en=getattr(item, "llm_title", None),
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
                    # Задача: пересобираем cluster_key с LLM-локацией.
                    for item, res in zip(fresh_batch, batch_results):
                        # сохраняем результат в item (всегда, даже ok=False)
                        item.llm_score = res.score
                        item.llm_summary = res.summary
                        item.llm_reason = res.reason
                        item.llm_title = getattr(res, "title_en", None)
                        item._batch_result = res
                        # Пересобираем cluster_key с LLM-локацией.
                        # Res.ok=False: LLM-ошибка → location='', rebuild даст 'other'.
                        # Это безопасно: cap не применяется к 'other'.
                        new_key = rebuild_cluster_key(
                            getattr(res, "location", "") or "",  # defensive
                            getattr(item, "cluster_key", None),
                            title=getattr(item, "title", None) or None,
                        )
                        item.cluster_key = new_key

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
            # push_cap: накапливаем push-eligible items, сортируем по score desc,
            # fetched_at asc (детерминированный порядок отбора), применяем лимит.
            # Top-N → push, остаток → deferred. При push_count_ref=None — штатный путь.
            _n_overflow: int = 0  # сверх-лимитные push-eligible items (для лога)
            if push_count_ref is not None:
                _push_eligible: List[tuple] = []  # (item, res, fetched_at)
                _pushed_urls: Set[str] = set()   # отслеживаем обработанные push-items
                for item in items_to_llm:
                    res = getattr(item, "_batch_result", None)
                    if res is None:
                        continue
                    # Critical: route() требует int score, ok=False даёт score=None → TypeError.
                    # Штатно ok=False обрабатывается ВНУТРИ _apply_result — вызываем напрямую.
                    # ok=False НЕ считается push → НЕ передаём ref (не инкрементируем счётчик).
                    if not res.ok:
                        self._apply_result(item, res, push_th, digest_th,
                                          push_urgent_only, skipped_urls, deferred_urls,
                                          to_deliver, llm_batch_records)
                        # FIX 3: добавляем url в processed-set → второй проход
                        # (:2389+) пропустит этот item (не двойной _apply_result).
                        _pushed_urls.add(item.url)
                        continue
                    # предварительная route — проверяем только push/digest/skip
                    r = route(res.score, push_th, digest_th,
                              urgent=res.urgent, push_urgent_only=push_urgent_only)
                    if r == "push":
                        # Fix 3: fetched_at может быть datetime или str — нормализуем в str
                        _fetched_at = getattr(item, "fetched_at", None)
                        if _fetched_at is None:
                            _fetched_at = ""
                        else:
                            # datetime → iso-строка для консистентного сравнения
                            _fetched_at = _fetched_at.isoformat() if hasattr(
                                _fetched_at, "isoformat"
                            ) else str(_fetched_at)
                        _push_eligible.append((item, res, _fetched_at))
                        item._batch_result = res  # уже установлен
                # Fix 5: сортировка только при активном cap (> 0)
                if max_pushes_per_cycle > 0:
                    # Сортировка: score desc (newer higher), при равном score —
                    # fetched_at asc (старые раньше). Детерминированный порядок отбора.
                    _push_eligible.sort(key=lambda x: (-(x[1].score or 0), x[2]))
                for _idx, (_item, _res, _fa) in enumerate(_push_eligible):
                    _is_overflow = max_pushes_per_cycle > 0 and push_count_ref[0] >= max_pushes_per_cycle
                    if _is_overflow:
                        _n_overflow += 1
                    # force_deferred=True → _apply_result отправляет в deferred без counter check
                    self._apply_result(_item, _res, push_th, digest_th,
                                       push_urgent_only, skipped_urls, deferred_urls,
                                       to_deliver, llm_batch_records,
                                       push_count_ref=push_count_ref,
                                       max_pushes_per_cycle=max_pushes_per_cycle,
                                       force_deferred=_is_overflow)
                    # counter инкрементируется ВНУТРИ _apply_result (когда not force_deferred)
                    _pushed_urls.add(_item.url)
                # Не-push items (digest/skip) — штатно, с тем же ref (счётчик не меняется)
                # Skip items already processed in the push-eligible loop above
                for item in items_to_llm:
                    res = getattr(item, "_batch_result", None)
                    if res is None:
                        continue
                    if item.url in _pushed_urls:
                        continue
                    if not res.ok:
                        # ok=False: штатная обработка, вызываем напрямую (route НЕ нужен).
                        # ok=False НЕ считается push → без ref чтобы не инкрементировать счётчик.
                        self._apply_result(item, res, push_th, digest_th,
                                          push_urgent_only, skipped_urls, deferred_urls,
                                          to_deliver, llm_batch_records)
                        continue
                    r = route(res.score, push_th, digest_th,
                              urgent=res.urgent, push_urgent_only=push_urgent_only)
                    if r != "push":
                        self._apply_result(item, res, push_th, digest_th,
                                          push_urgent_only, skipped_urls, deferred_urls,
                                          to_deliver, llm_batch_records,
                                          push_count_ref=push_count_ref,
                                          max_pushes_per_cycle=max_pushes_per_cycle)
            else:
                # Без ref — штатная обработка (без push-cap)
                for item in items_to_llm:
                    res = getattr(item, "_batch_result", None)
                    if res is None:
                        continue
                    self._apply_result(item, res, push_th, digest_th,
                                       push_urgent_only, skipped_urls, deferred_urls,
                                       to_deliver, llm_batch_records)

            # сохраняем все llm-результаты батчем (push + deferred + skipped)
            # Per-record изоляция: возвращается список успешно обновлённых URL.
            # set_state вызывается ТОЛЬКО для них — сбойные остаются pending.
            updated_urls: Set[str] = set()
            if llm_batch_records:
                updated_urls = set(self.storage.update_llm_results_batch(llm_batch_records))

            # ── фиксируем state и логируем итоговые счётчики (одна INFO-строка) ──
            # Minor #1: вычисляем пересечение один раз, переиспользуем.
            actually_skipped = list(skipped_urls & updated_urls)
            actually_deferred = list(deferred_urls & updated_urls)
            n_skipped = len(actually_skipped)
            n_deferred = len(actually_deferred)
            n_push = len(to_deliver)

            # set_state только для реально обновлённых URL
            # (провалившийся UPDATE → item в pending, его подберёт recovery)
            if actually_skipped:
                self.storage.set_state(actually_skipped, "skipped")
            if actually_deferred:
                self.storage.set_state(actually_deferred, "deferred")

            mode_tag = " [urgent-only]" if push_urgent_only else ""
            # overflow-log: только когда cap активен (max>0) и ref есть
            if max_pushes_per_cycle > 0 and push_count_ref is not None and _n_overflow > 0:
                logger.info(
                    "Маршрутизация%s: push=%d deferred=%d skipped=%d [%d push сверх лимита %d → дайджест]",
                    mode_tag, n_push, n_deferred, n_skipped, _n_overflow, max_pushes_per_cycle,
                )
            else:
                logger.info(
                    "Маршрутизация%s: push=%d deferred=%d skipped=%d",
                    mode_tag, n_push, n_deferred, n_skipped,
                )
            # Major-3: prefilter-статистика логируется в run_cycle_with_delivery
            # (до LLM-дедупа). Здесь только маршрутизация и LLM-результаты.

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
            delivered_urls, permanent_failed_urls = self.delivery.send_batch(to_deliver)
            if delivered_urls:
                self.storage.set_state(delivered_urls, "delivered")
            # FIX-1: permanent_failed_urls → state='failed' напрямую (без retry_count).
            # По образцу mark_failed_if_retries_exhausted / set_state('failed')
            # в существующем коде (pipeline.py:2088, 2224, 2283).
            if permanent_failed_urls:
                self.storage.set_state(permanent_failed_urls, "failed")
                logger.info(
                    "Доставлено %d/%d items; %d permanent-failed (→ state='failed')",
                    len(delivered_urls), len(to_deliver), len(permanent_failed_urls),
                )
            perm_set = set(permanent_failed_urls)
            not_delivered = [i.url for i in to_deliver
                             if i.url not in delivered_urls
                             and i.url not in perm_set]
            if not_delivered:
                logger.info(
                    "Доставлено %d/%d items; %d не доставлено (pending → retry)",
                    len(delivered_urls), len(to_deliver), len(not_delivered),
                )
            elif not permanent_failed_urls:
                logger.info(
                    "Доставлено %d/%d items",
                    len(delivered_urls), len(to_deliver),
                )
        except (ChatIdNotBoundError, BotBlockedError) as e:
            # Critical-1-фикс: при partial success (BatchAbortedError) читаем
            # successful_urls / permanent_failed_urls — помечаем РЕАЛЬНО доставленные
            # и терминализируем permanent-failed. Без этого дубликаты уходили в
            # повторную отправку.
            succ = getattr(e, "successful_urls", []) or []
            perm = getattr(e, "permanent_failed_urls", []) or []
            if succ:
                self.storage.set_state(succ, "delivered")
                logger.info(
                    "Частичная доставка: %d items → 'delivered', %d → 'failed' "
                    "(по %s)",
                    len(succ), len(perm), type(e).__name__,
                )
            if perm:
                self.storage.set_state(perm, "failed")
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
        skipped_urls: Set[str],
        deferred_urls: Set[str],
        to_deliver: List["NewsItem"],
        llm_batch_records: List[dict],
        push_count_ref: Optional[List[int]] = None,
        max_pushes_per_cycle: int = 0,
        force_deferred: bool = False,
    ) -> None:
        """
        Применяет FilterResult к одному item: маршрутизация + сбор record.
        Вызывается из _process_batch после evaluate / synthesize (retry).
        """
        if not res.ok:
            # Major-4: fatal-ошибка (401/402/403) — как cost_blocked: item остаётся
            # pending, НЕ уходит в fallback/deferred/push.
            # Защитный getattr: работает и без поля fatal (до параллельного PR).
            if getattr(res, "fatal", False):
                logger.debug(
                    "fatal: '%s' остаётся pending (LLM-авторизация недоступна)",
                    item.url,
                )
                return
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
                deferred_urls.add(item.url)
                llm_batch_records.append({
                    "url": item.url,
                    "score": None,
                    "summary": "",
                    "reason": "",
                    "urgent": False,
                    "cluster_key": getattr(item, "cluster_key", None),
                    "title_en": getattr(res, "title_en", None),
                })
            else:
                to_deliver.append(item)
                item._feedback_kind = "fallback"  # type: ignore[attr-defined]
            return

        r = route(res.score, push_th, digest_th,
                  urgent=res.urgent, push_urgent_only=push_urgent_only)
        # Minor #3: если LLM вернул пустой summary, а RSS summary есть —
        # используем его как fallback, но помечаем llm_reason, чтобы
        # отличить от осознанно пустого ответа.
        rss_summary = getattr(item, "summary", "") or ""
        used_fallback = not res.summary and rss_summary
        if used_fallback:
            reason = (res.reason or "no_summary") + "|phase1_only"
        else:
            reason = res.reason or "no_summary"
        final_summary = res.summary or rss_summary
        record = {
            "url": item.url,
            "score": res.score,
            "summary": final_summary,
            "reason": reason,
            "urgent": res.urgent,
            "cluster_key": getattr(item, "cluster_key", None),
            "title_en": getattr(res, "title_en", None),
        }
        if r == "skip":
            skipped_urls.add(item.url)
        elif r == "digest":
            deferred_urls.add(item.url)
        else:
            # r == "push"
            # Push-cap: если лимит достигнут и это не принудительный deferred →
            # принудительно уводим overflow в дайджест.
            if force_deferred:
                deferred_urls.add(item.url)
            else:
                _ok_to_push = True
                if max_pushes_per_cycle > 0 and push_count_ref is not None:
                    # Cap active: проверяем и инкрементируем
                    if push_count_ref[0] >= max_pushes_per_cycle:
                        deferred_urls.add(item.url)
                        _ok_to_push = False
                    else:
                        push_count_ref[0] += 1
                elif push_count_ref is not None:
                    # Unlimited (cap=0): считаем pushи (для отладки/тестов)
                    push_count_ref[0] += 1
                if _ok_to_push:
                    to_deliver.append(item)
                    item._feedback_kind = "push"  # type: ignore[attr-defined]
        llm_batch_records.append(record)

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
        poll_interval_minutes = _coerce_int(
            self.config.get("poll_interval_minutes", 15), 15, "poll_interval_minutes",
        )
        grace_delta = timedelta(minutes=poll_interval_minutes)

        # ── Оконная логика: граница окна ───────────────────────────────────
        # Задача 1: слотовое окно с якорем.
        # digest.window_anchor = "HH:MM" (локальное время) — граница окна
        # привязана к якорю (последний наступивший до digest_started).
        # Anchor-режим: published-фильтр ВСЕГДА активен; откат к
        # unprocessed НЕ применяется; граница фиксирована якорем.
        # Fallback (ключ отсутствует/битый): прежнее поведение.
        now_utc = datetime.now(tz.utc).replace(microsecond=0)
        digest_started_iso = now_utc.isoformat()
        anchor_raw = digest_cfg.get("window_anchor")
        anchor_mode = False
        if anchor_raw and isinstance(anchor_raw, str):
            try:
                anchor_parsed = _time.strptime(anchor_raw.strip(), "%H:%M")
                anchor_hour = anchor_parsed.tm_hour
                anchor_min = anchor_parsed.tm_min
                anchor_mode = True
            except (TypeError, ValueError):
                logger.warning(
                    "digest.window_anchor=%r невалидно — использую fallback (meta)",
                    anchor_raw,
                )
        raw_boundary = self.storage.get_meta("digest_window_started_at")
        if anchor_mode:
            # Anchor-режим: граница = ближайший предшествующий якорь (вчера).
            # Слот 20:00 (digest_started=20:00 сегодня) → вчера 20:00 (24ч окно).
            # Слот 08:00 (digest_started=08:00 сегодня) → вчера 20:00 (12ч окно).
            #
            # Алгоритм: yesterday_anchor = сегодня 00:00 local - 1 day + anchor_time.
            # Пример UTC+3, now_local=Oct22 11:00:
            #   midnight local (Oct22 00:00 UTC+3 = Oct21 21:00 UTC)
            #   - 1 day → Oct21 00:00 UTC+3 = Oct20 21:00 UTC
            #   + anchor 20:00 → Oct21 20:00 UTC+3 = Oct21 17:00 UTC.
            now_local = _local_now().replace(microsecond=0)
            today_midnight_local = now_local.replace(
                hour=0, minute=0, second=0, microsecond=0,
            )
            yesterday_midnight_local = today_midnight_local - timedelta(days=1)
            window_start_local = yesterday_midnight_local.replace(
                hour=anchor_hour, minute=anchor_min, second=0, microsecond=0,
            )
            window_start_iso = window_start_local.astimezone(tz.utc).isoformat()
            # FIX 4: min(anchor, meta_boundary) — берём более СТАРУЮ/широкую границу.
            # Если digest_slot был пропущен, meta boundary осталась от
            # предыдущего окна и более свежая (=более узкая). min() берёт её,
            # чтобы покрыть items из пропущенного слота.
            if raw_boundary:
                from datetime import datetime as dt, timezone as tz_dummy
                try:
                    meta_dt_raw = dt.fromisoformat(raw_boundary.replace('Z', '+00:00'))
                    if meta_dt_raw.tzinfo is None:
                        # naive — считаем UTC
                        meta_dt = meta_dt_raw.replace(tzinfo=tz_dummy.utc)
                    else:
                        meta_dt = meta_dt_raw.astimezone(tz_dummy.utc)
                    anchor_dt = dt.fromisoformat(window_start_iso.replace('Z', '+00:00'))
                    if anchor_dt.tzinfo is None:
                        anchor_dt = anchor_dt.replace(tzinfo=tz_dummy.utc)
                    # FIX 4/F: min(anchor, meta) покрывает пропущенный слот.
                    # Freshness guard: meta применяется только если она не старше
                    # bootstrap_window_hours ОТ anchor (а не от now). Это защищает
                    # от слишком старых meta (ручные правки, миграция, пропущенный
                    # слот давно в прошлом) — они игнорируются, окно берётся от anchor.
                    bootstrap_hours = digest_cfg.get("bootstrap_window_hours", 24)
                    try:
                        bootstrap_hours = float(bootstrap_hours)
                    except (TypeError, ValueError):
                        bootstrap_hours = 24.0
                    # FIX F: freshness measured from anchor, not from now.
                    # stale_if: meta_dt < anchor_dt - bootstrap_hours → игнорировать meta.
                    if meta_dt >= anchor_dt - timedelta(hours=bootstrap_hours):
                        window_start_iso = (meta_dt if meta_dt < anchor_dt else anchor_dt).isoformat()
                except (TypeError, ValueError):
                    pass  # fallback к anchor
            if not raw_boundary:
                logger.info(
                    "Дайджест (anchor=%s): bootstrap → %s",
                    anchor_raw, window_start_iso,
                )
            else:
                logger.debug(
                    "Дайджест (anchor=%s): граница из meta, anchor=%s",
                    anchor_raw, window_start_iso,
                )
        elif not raw_boundary:
            # Fallback bootstrap
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
            # Anchor-режим: always use window_start_iso (including bootstrap).
            # Fallback: only if boundary already set; bootstrap — no SQL filter
            # (published-фильтр тоже disabled at Python level).
            since_iso=window_start_iso if (anchor_mode or raw_boundary is not None) else None,
            # Anchor-режим: max_age_hours НЕ применяется — граница окна
            # определяется якорем. Слот 20:00 при запуске в 20:05 (scheduler lag)
            # не должен терять первые минуты окна (bound = yesterday 20:00, не
            # min(yesterday 20:00, now-24h) = now-24h < yesterday 20:00 → пропуск).
            # Fallback: прежнее поведение (max_age_hours ограничивает выборку).
            # Decision: pass None in anchor mode — минимальный diff, storage
            # already correctly handles None.
            max_age_hours=max_age_hours if not anchor_mode else None,
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
        # Задача 1 (п.4): published-фильтр ВСЕГДА активен в anchor-режиме.
        # Fallback: активен только если граница уже установлена (raw_boundary
        # is not None); при bootstrap (raw_boundary is None) — published-фильтр
        # отключён, чтобы items с непарсящимся published не терялись.
        published_filter_active = anchor_mode or (raw_boundary is not None)
        ws_dt = _iso_dt(window_start_iso)
        if published_filter_active and ws_dt is not None:
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
                    if pub_dt_norm > ws_grace_naive:
                        filtered.append(it)
            deferrable_post_published = filtered
        n_published = len(deferrable_post_published)

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

        _n_dropped = len(deferrable_post_published) - n_top_n
        logger.info(
            "Дайджест: старт — newsbot v%s, deferred=%d, dropped=%d, total=%d",
            __version__, len(deferred_items), _n_dropped, n_published,
        )

        # M2-фикс: при пустом deferred + on_empty=skip пишем маркер ОКНА,
        # чтобы _maybe_run_digest не спамил в следующем цикле.
        # Оконная логика: пустое окно → сдвигаем границу.
        if not deferred_items and not force:
            if on_empty == "skip":
                # Minor 1-фикс: dropped-счётчик через count_deferred_before
                # (единообразно с главной веткой), а не overflow от срезки.
                _n_dropped = self.storage.count_deferred_before(window_start_iso)
                logger.info(
                    "Дайджест: пустое окно (on_empty=skip) — dropped=%d, total=%d",
                    _n_dropped, n_published,
                )
                logger.debug("Дайджест: нет отложенных новостей, окно проверено (skip)")
                self.storage.set_meta(
                    "last_digest_at",
                    digest_started_iso,
                )
                # Задача 1 (п.5): anchor-режим — граница фиксирована якорем.
                # Fallback — прежний rollback к unprocessed.
                if anchor_mode:
                    new_boundary = window_start_iso
                else:
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
        _dup_urls, _recent_shown = self._cluster_deferred_duplicates(deferred_items, window_start_iso)
        if _dup_urls:
            deferred_items = [it for it in deferred_items if it.url not in _dup_urls]
        n_dedup = len(deferred_items)

        # ── F6: детерминированная страховка после LLM-кластеризации ──────────
        # LLM мог пропустить containment-пары, которые не попали в его контекст.
        # Финальный проход: для всех пар кандидатов — containment-check
        # (≥30 симв., числовое равенство, Jaccard ≥0.8, word-boundary containment).
        # Лидер — первый по текущей сортировке score DESC, fetched_at ASC.
        # Дубли помечаются в _containment_dup_urls → исключаются из отправки.
        _containment_dup_urls: Set[str] = set()
        if len(deferred_items) >= 2:
            # Текущий порядок: SQL (score DESC, fetched_at ASC) от LLM-дедупа.
            # items[i] — потенциальный лидер для items[j] (i < j).
            for i in range(len(deferred_items)):
                a = deferred_items[i]
                if a.url in _containment_dup_urls:
                    continue
                for j in range(i + 1, len(deferred_items)):
                    b = deferred_items[j]
                    if b.url in _containment_dup_urls:
                        continue
                    a_title = getattr(a, 'title_normalized', None)() if hasattr(a, 'title_normalized') else ""
                    b_title = getattr(b, 'title_normalized', None)() if hasattr(b, 'title_normalized') else ""
                    if not a_title or not b_title:
                        continue
                    icd = is_containment_duplicate(a_title, b_title)
                    if icd:
                        logger.info(
                            "F6 containment: dup %s (%s) отсеян лидером %s (%s)",
                            getattr(b, 'url', ''), b_title[:60],
                            getattr(a, 'url', ''), a_title[:60],
                        )
                        _containment_dup_urls.add(b.url)
            n_containment = len(_containment_dup_urls)
            if n_containment:
                deferred_items = [it for it in deferred_items if it.url not in _containment_dup_urls]
                logger.info(
                    "Дайджест: containment-страховка отсеяла %d дублей из %d "
                    "(%d кандидатов после LLM-дедупа)",
                    n_containment, n_containment + len(deferred_items), n_dedup,
                )
                _dup_urls = _dup_urls | _containment_dup_urls
        n_dedup = len(deferred_items)

        # Задача 2 (M1): дубли на границе max_items.
        # _check_against_shown (ранний backfill :2213, skipped :2281) видит только
        # digested/delivered из _recent_shown (get_shown_for_window).
        # Но item X в top-N + его пересказ Y в хвосте deferred-пула
        # НИГДЕ не сравниваются → оба уходят в дайджест.
        # Фикс: включаем title/source уже отобранных top-N items в RECENT-контекст.
        # Формат — тот же, что у get_shown_for_window: [{title, source}].
        # dedup (set) уже отфильтровал лидеров — берём текущий deferred_items
        # (post-dedup лидеры, самые приоритетные).
        _digest_topn_shown: List[dict] = [
            {"title": it.title, "source": it.source}
            for it in deferred_items
        ]
        _seen_keys: Set[tuple] = set()
        for sh in _recent_shown:
            _seen_keys.add((sh.get("title", ""), sh.get("source", "")))
        for sh in _digest_topn_shown:
            _key = (sh.get("title", ""), sh.get("source", ""))
            if _key not in _seen_keys:
                _recent_shown.append(sh)
                _seen_keys.add(_key)

        # ── Задача A: бэкфилл после LLM-дедупа ──────────────────────────────
        # Если в топе оказались дубли — дайджест недобирается (буфер max_items..2*max_items
        # не используется). Добираем из pool_tail: items, изначально БЫВШИХ ЗА
        # пределами top-N (deferrable_post_published[max_items:]). Важно: items,
        # которые были В top-N, но были удалены dedup-ом, НЕ берутся из пула —
        # они уже прошли top-N и не являются «запасными» кандидатами.
        #
        # MAJOR-фикс (4c): бэкфилл items проходят RECENT-дедуп перед добавлением
        # в дайджест (всё попадающее в дайджест должно быть проверено против
        # показанных). Запасные items прошли pre-LLM cluster_key caps и
        # domain-дедуп, но не LLM-дедуп против показанных.
        self._backfill_dup_urls: Set[str] = set()
        if deferrable_post_published and max_items and max_items > 0:
            pool_tail = deferrable_post_published[max_items:]
            backfill_added = 0
            existing_urls = {x.url for x in deferred_items}
            # MAJOR-фикс (4c): фильтруем пул против показанных перед добавлением.
            # _check_against_shown использует уже загруженный _recent_shown.
            self._backfill_dup_urls = self._check_against_shown(pool_tail, _recent_shown)
            for it in pool_tail:
                if len(deferred_items) >= max_items:
                    break
                if it.url not in existing_urls and it.url not in self._backfill_dup_urls:
                    deferred_items.append(it)
                    existing_urls.add(it.url)
                    backfill_added += 1
            if backfill_added:
                logger.info(
                    "Дайджест: бэкфилл добавил %d items из пула (top-N был неполным)",
                    backfill_added,
                )
            elif self._backfill_dup_urls:
                logger.debug(
                    "Дайджест: бэкфилл пропущен (%d дублей против показанных)",
                    len(self._backfill_dup_urls),
                )
        n_backfill = len(deferred_items)

        # ── Задача B: гарантия по категориям с возможностью ПДТЯГИВАНИЯ ──────
        # Pool для гарантии: deferrable_post_published минус всё, что уже в
        # deferred_items по url (id не используем — cap теперь после гарантии).
        # MAJOR: cap by location/day — строго ПОСЛЕ _category_guarantee.
        # Передвинут в конец, чтобы инвариант «≤max_per_location на (локация, день)»
        # держался железно даже после свапов гарантии.
        #
        # Задача 2: добор из skipped. Skipped-items (score < digest_threshold
        # но >= guarantee_min_score) не попали в deferred, но имеют llm_summary
        # (скорились) и подходят для гарантии кино. Добавляем их в category_pool
        # ПОСЛЕ deferred-pool: они — последний резерв.
        category_pool: List[NewsItem] = []
        # MAJOR-фикс (4c): items, отклонённые backfill-дедупом, не должны
        # попадать в guarantee.
        included_urls = {it.url for it in deferred_items}
        _all_excluded = included_urls | _dup_urls | self._backfill_dup_urls
        if deferrable_post_published:
            category_pool = [
                it for it in deferrable_post_published
                if it.url not in _all_excluded
            ]
        # Skipped: порог guarantee_min_score (default 3 из конфига).
        # Minor 1-фикс: гарантия кино падала при невалидном guarantee_min_score
        # (строка/список в YAML → ValueError/TypeError). По конвенции проекта —
        # WARNING + documented default 3.
        try:
            guarantee_min_score = int(digest_cfg.get("guarantee_min_score", 3))
        except (TypeError, ValueError):
            guarantee_min_score = 3
            logger.warning(
                "digest.guarantee_min_score=%r невалидно — использую 3",
                digest_cfg.get("guarantee_min_score"),
            )
        # MAJOR-фикс (4c): skipped-кандидаты из БД тоже должны пройти
        # RECENT-дедуп. О утренней новости, показанной в утреннем слоте,
        # может попасть в skipped → вечерний guarantee может добавить её
        # повторно. Проверяем _recent_shown.
        # Minor 4-фикс: лимит 50 → 200. Кино может остаться «голодным» при
        # ограниченном skip-pool. SQL сортирует llm_score DESC — гарантия
        # возьмёт лучшие cinema-items из большего пула. Альтернатива: фильтр
        # по cinema-источникам в SQL — сложнее, не обязательно.
        category_min = _normalize_category_min(digest_cfg.get("category_min", {}))
        if category_min:
            skipped_candidates = self.storage.get_skipped_items(
                since_iso=window_start_iso,
                min_score=guarantee_min_score,
                limit=200,
            )
            _skipped_dup_urls = self._check_against_shown(skipped_candidates, _recent_shown)
            for it in skipped_candidates:
                if it.url not in _all_excluded and it.url not in _skipped_dup_urls:
                    category_pool.append(it)
            # Fallback: нескоренные (state='new') из того же окна — последний резерв.
            # Передаются отдельным параметром в _category_guarantee (new_fallback),
            # чтобы скоренные (pool, score DESC) всегда были приоритетнее new
            # (score=NULL → 0). _check_against_shown: новые items не были
            # в digested/delivered → не в _recent_shown → не фильтруются.
            # Major-фикс: new-кандидаты проходят префильтр — мусор/реклама
            # (напр. off-peak-пауза без LLM в цикле) не попадает в гарантию.
            _raw_new = self.storage.get_new_items_for_fallback(
                since_iso=window_start_iso,
                limit=max(200, max_items * 4),
            )
            _new_candidates = [it for it in _raw_new if not _prefilter_item(it)]
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
        guarantee_added_urls: Set[str] = set()
        if category_min:
            deferred_items, guarantee_added_urls = _category_guarantee(
                deferred_items, category_min, pool=category_pool,
                new_fallback=_new_candidates if category_min else None,
            )
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
            # Задача 3 (M2): поздний backfill пропускает url из self._backfill_dup_urls.
            # self._backfill_dup_urls вычисляется в раннем backfill (:2238),
            # может быть пустым множеством если ранний backfill не выполнялся
            # (early return по пустому deferrable_post_published или max_items<=0).
            for it in deferrable_post_published:
                if it.url in included_urls:
                    continue
                # M1: пропускаем LLM-дубли (не были в deferred_items после dedup,
                # но они в deferrable_post_published)
                if it.url in _dup_urls:
                    continue
                # Задача 3 (M2): пропускаем url, отклонённые ранним backfill
                # как дубли показанных — иначе поздний backfill возвращает их.
                if it.url in self._backfill_dup_urls:
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
        # Critical-фикс (wave 2): резервируем слоты под ВСЕ защищённые items.
        # guarantee_added_urls защищает только ДОБАВЛЕННЫЕ гарантией.
        # guarantee_protected_urls ⊇ guarantee_added_urls: включает ТАКЖЕ исходные
        # deferred-items, которые САМИ закрывают часть category_min.
        # Без этого исходный cinema-item с низким score вырезается [:max_items]
        # → категория ниже минимума, несмотря на new-fallback.
        # Используем guarantee_protected_urls — минимальный корректный надмножество.
        _protected_item_ids: Set[int] = set()
        for _it in deferred_items:
            _key = getattr(_it, 'url', None) or id(_it)
            if _key in guarantee_protected_urls:
                _protected_item_ids.add(id(_it))
        _non_protected = [
            _it for _it in deferred_items if id(_it) not in _protected_item_ids
        ]
        _non_protected.sort(key=lambda x: x.llm_score or 0, reverse=True)
        _slots = max(0, max_items - len(_protected_item_ids))
        _final = _non_protected[:_slots]
        _final_urls = {getattr(_it, 'url', None) for _it in _final}
        for _it in deferred_items:
            if id(_it) in _protected_item_ids:
                if getattr(_it, 'url', None) not in _final_urls:
                    _final.append(_it)
        deferred_items = _final[:max_items]
        n_final = len(deferred_items)

        # F: наблюдаемость — DEBUG-дамп финального состава.
        # Показывает url/title/location/cluster_key первых min(N, 10) items.
        if logger.isEnabledFor(logging.DEBUG):
            _dbg_lines = []
            for _idx, _it in enumerate(deferred_items[:10]):
                _ck = getattr(_it, 'cluster_key', '') or ''
                _loc = _ck.split('|')[0] if _ck else 'no-key'
                _dbg_lines.append(
                    f"  [{_idx}] score={getattr(_it, 'llm_score', '?')} "
                    f"loc={_loc!r} url={getattr(_it, 'url', '?')}"
                )
            _dbg_msg = (
                f"Дайджест DEBUG (final {n_final} items):\n"
                + "\n".join(_dbg_lines)
                + (f"\n  ... и ещё {max(0, n_final - 10)} items" if n_final > 10 else "")
            )
            logger.debug(_dbg_msg)

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
                    # Minor 1-фикс: терминализируем дубли ДО возврата.
                    # ⚠️ Осознанный риск: при false-positive containment (LLM ошибся,
                    # два unrelated events попали в один кластер) оба получат
                    # state='digested' и пропадут — это терминальное состояние.
                    if _dup_urls:
                        try:
                            self.storage.set_digested(list(_dup_urls))
                        except Exception:
                            pass
                    return 0
            # Minor 1-фикс: весь дайджест — дубли (легально при непустом recent).
            # Без set_digested дубли вечно deferred → повторный расход токенов
            # на _cluster_deferred_duplicates в каждом окне.
            if _dup_urls:
                try:
                    self.storage.set_digested(list(_dup_urls))
                except Exception:
                    pass
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
            # FIX 5: 3-strike сдвигает границу окна СОВМЕСТНО со слотом.
            # Если сдвигать только last_digest_at (слот обработан), но не
            # digest_window_started_at — следующий слот считает границу от
            # anchor и отсечёт published-фильтром (:2908) все items старше
            # этой границы. FIX 5 обновляет ОБЕ мета-поля → следующий слот
            # получает корректное покрытие. При shutdown — last_digest_at
            # НЕ пишется, boundary не двигается.
            sample_urls = [it.url for _, grp in groups for it in grp][:5]
            logger.warning(
                "Дайджест: пропущен (%d неудачных попыток подряд) — "
                "%d items остаются deferred, граница сдвинута, "
                "счётчик сброшен; примеры url: %s",
                failure_count, total_to_send, ", ".join(sample_urls) or "—",
            )
            self.storage.set_meta("digest_failure_count", "0")
            self.storage.set_meta("last_digest_at", digest_started_iso)
            # FIX 5: сдвигаем границу окна вместе со слотом.
            # Раньше last_digest_at обновлялся (слот "обработан"), а
            # digest_window_started_at — нет → следующий слот получал
            # 12-часовое окно от anchor, items старше границы отсекались
            # published-фильтром (:2908) → тихая потеря целого окна.
            # Обновляем ОБЕ мета-поля синхронно.
            self.storage.set_meta("digest_window_started_at", digest_started_iso)
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
            # M3-фикс: run_digest возвращает 4-tuple
            # (sent_count, sent_urls, digest_failed_urls, _permanent_urls).
            # _permanent_urls НЕ нужны здесь — digest.run_digest уже объединил
            # их с failed_urls_in в digest_failed_urls.
            sent_count, sent_urls, digest_failed_urls, _ = _run_digest(
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
                    "digest: %d/%d items → 'failed' (рендеринг / "
                    "feedback-target / permanent delivery)",
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
            # Задача 1 (п.5): anchor-режим — граница фиксирована якорем.
            # Fallback — прежний rollback к unprocessed.
            # Задача 1 (п.7): лог сброшенных deferred-items.
            dropped_count = self.storage.count_deferred_before(window_start_iso)
            if dropped_count:
                logger.warning(
                    "Дайджест: %d deferred items dropped by window close "
                    "(fetched_at < %s)",
                    dropped_count, window_start_iso,
                )
            if anchor_mode:
                new_boundary = window_start_iso
            else:
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
                # Anchor: граница фиксирована. Fallback: rollback к unprocessed.
                dropped_count = self.storage.count_deferred_before(window_start_iso)
                if dropped_count:
                    logger.warning(
                        "Дайджест: %d deferred items dropped by window close "
                        "(fetched_at < %s)",
                        dropped_count, window_start_iso,
                    )
                if anchor_mode:
                    new_boundary = window_start_iso
                else:
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

        # Нормализация в точке чтения: невалидные слоты отфильтровываются
        # с warning, не-список → дефолт; ни один слот не должен ронять
        # _maybe_run_digest (TypeError вне try → дайджест не запускался бы).
        slots = _normalize_digest_times(digest_cfg.get("times"))
        if not slots:
            return

        now_local = _local_now()
        last_raw = self.storage.get_meta("last_digest_at")
        last_digest: datetime | None = None
        if last_raw:
            try:
                last_digest = datetime.fromisoformat(last_raw).astimezone()
            except Exception:
                pass

        for sched_hour, sched_min in slots:
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