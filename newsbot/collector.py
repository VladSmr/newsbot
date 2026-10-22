"""
Коллектор RSS-фидов: feedparser + httpx.
Один фид не роняет цикл.

Корректность:
  - resp.content (bytes) → feedparser.parse() — автодекодинг по XML Declaration
  - published_parsed → calendar.timegm (UTC) вместо time.mktime (localtime)
  - пустые entry без link пропускаются
"""
import calendar
import html as _html
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, List, Optional, Pattern, Tuple

import feedparser
import httpx

logger = logging.getLogger("newsbot.collector")

# ─── Лимиты ──────────────────────────────────────────────────────────────────
# Wall-clock (точнее — monotonic) deadline на ОДИН фид целиком:
# HTTP-фаза (подключение + чтение тела) И конвертация entries.
# Ставится ДО self._client.stream() — иначе медленный сервер, отдающий
# по байту, держит iter_bytes() неограниченно долго (per-operation
# httpx-таймаут сбрасывается на каждом чанке).
FEED_DEADLINE_SEC = 30
MAX_ITEMS_PER_FEED = 50  # защита от кривого фида

# ─── Префиксы источников для стриппинга ───────────────────────────────────────
# Стриппаются ТОЛЬКО в начале строки, ТОЛЬКО если после них идёт whitespace
# и далее есть осмысленный текст. Не трогают заголовки с ":" или "—" в середине.
# Формат: (compiled_pattern, replacement)
_SOURCE_PREFIXES: List[Tuple[Pattern[str], str]] = [
    # "Reuters: Headline", "Bloomberg: Story", "Источник: Что-то произошло"
    (re.compile(r"^\s*(?:Reuters|Bloomberg|TASS|Reuters:|Bloomberg:|TASS:|"
                r"Источник|Al\s?Jazeera|RBC|Kommersant|Interfax|РИА\s?Новости|"
                r"Лента\.ру|Eurogamer|Guardian|FRANCE\s?24|Sky\s?News|AeroTime|"
                r"NASA|Naked\sScience|N\+1|STAT\sNews|Phys\.org|ScienceDaily|"
                r"BleepingComputer|KrebsOnSecurity|Habr|Tom['\u2019]?s\sHardware|"
                r"SpaceNews|NASASpaceflight|IGN|DTF|Deadline|Variety|"
                r"Югополис|Кубанские\sновости)\s*[:—–-]+\s*", re.IGNORECASE), ""),
    # "Al Jazeera — Headline", "Bloomberg — News"
    (re.compile(r"^\s*(?:Al\s+Jazeera|Bloomberg|REUTERS|AP|AFP)\s*[—–-]+\s+",
                 re.IGNORECASE), ""),
    # "From Source: Headline", "From Source A — Story" (RSS syndication pattern)
    # NOTE: разделитель [:—–-]+ — ОБЯЗАТЕЛЬНЫЙ (не *), чтобы "From Source to Sea…"
    # не срезалось; \s+ после него гарантирует наличие текста после префикса.
    # Регекс-класс [:—–-]: «:» (U+003A), EM-dash (U+2014), EN-dash (U+2013),
    # ASCII '-' (U+002D). NBSP (U+00A0) УДАЛЕНА: она стояла между EM-dash и '-',
    # что в character class трактовалось как диапазон U+00A0–U+002D, включающий
    # весь ASCII range 0xA0–0x2D, а '-' (U+002D) сам при этом НЕ матчился.
    # Исправление: '-' в конец символьного класса — не трактуется как range operator.
    (re.compile(r"^\s*From\s+Source\s*[:—–-]+\s+", re.IGNORECASE), ""),
]

_NON_WORD_CHARS = re.compile(r"[^\w\s]")
_MULTI_SPACE = re.compile(r"\s+")


def _strip_source_prefix(title: str) -> str:
    """
    Срезает типовой префикс источника в начале строки.
    Безопасно: только начало строки, только известные паттерны,
    требует whitespace после префикса → не ломает заголовки с ":" в середине.
    """
    result = title
    for pattern, replacement in _SOURCE_PREFIXES:
        result = pattern.sub(replacement, result)
    return result

FETCH_TIMEOUT = 15  # секунд
MAX_FEED_BYTES = 10 * 1024 * 1024  # 10 MB — защита от gzip-бомбы / гигантского фида
USER_AGENT = (
    "Mozilla/5.0 (compatible; NewsBot/1.0; +https://github.com/newsbot) "
    "Python/3.8/httpx"
)

# ─────────────────────────────────────────────────────────────────
# Dataclass — опциональные поля под результат LLM (этап 2)
# ─────────────────────────────────────────────────────────────────


@dataclass
class NewsItem:
    """Единица новости из фида."""

    source: str
    url: str
    title: str
    summary: str
    published: str
    tags: Optional[str] = None                # JSON-массив тегов из <category>
    published_parsed: Optional[datetime] = None
    cluster_key: Optional[str] = None         # локация|stem_hash|день (для кластеризации)
    lang: Optional[str] = None                # язык фида (ru/en), проводка из config.yaml → БД → prompt
    # --- результат LLM-фильтра (этап 2) ---
    llm_score: Optional[int] = None          # оценка 0..10
    llm_summary: Optional[str] = None       # выжимка от LLM
    llm_reason: Optional[str] = None         # краткое пояснение
    # C1: llm_urgent — персистентный флаг срочности (INTEGER 0/1 в БД, v10).
    # None при первом LLM-вызове; заполняется после evaluate + update_llm_results_batch.
    # Retry: восстанавливается из БД через _rows_to_items → retry сохраняет срочность.
    # Legacy-строки (до v10) имеют llm_urgent=NULL → расценивается как False.
    llm_urgent: Optional[bool] = None
    retry_count: int = 0                     # счётчик попыток retry
    fetched_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # --- Фаза 2b: transient runtime-поле для кнопок фидбэка ---
    # Не является колонкой БД; заполняется в pipeline перед отправкой push-сообщения.
    feedback_target_id: Optional[int] = None

    def title_normalized(self) -> str:
        """
        Нормализованный заголовок для дедупа:
          1. Срезать типовой префикс источника (Reuters:, Bloomberg — и т.п.)
          2. lowercase
          3. Удалить пунктуацию / не-буквенные символы
          4. Свернуть повторные пробелы и strip.

        Семантика: title_hash = title_normalized(), используется в seen_titles.
        При изменении нормализации (TTL 3 дня) — старые title_hash остаются
        с прежней семантикой до истечения TTL. Это приемлемо (TTL 3 дня).
        """
        # (1) стриппинг префикса источника
        step1 = _strip_source_prefix(self.title)
        # (2) lowercase
        step2 = step1.lower()
        # (3) удалить пунктуацию / не-буквенные
        step3 = _NON_WORD_CHARS.sub("", step2)
        # (4) collapse пробелы + strip
        step4 = _MULTI_SPACE.sub(" ", step3).strip()
        return step4


# ─────────────────────────────────────────────────────────────────
# Collector
# ─────────────────────────────────────────────────────────────────


class FeedResult:
    """
    Результат одного фида.
    ok=True  — фид загружен и распарсен, items может быть пустым списком
    ok=False — фид упал, items = []
    """

    __slots__ = ("ok", "items")

    def __init__(self, ok: bool, items: List[NewsItem]):
        self.ok = ok
        self.items = items


class FeedCollector:
    """Последовательный обход фидов. Ошибка одного — логируется, цикл продолжается."""

    def __init__(self, sources: List[dict]):
        self.sources = sources
        self._client = httpx.Client(
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            trust_env=True,  # подхватывает HTTP_PROXY / HTTPS_PROXY из окружения (Docker)
        )

    def _validate_url(self, url: str) -> bool:
        """
        M3-фикс: проверяет URL до HTTP-запроса (SSRF-защита).
        Принимаются только http:// и https:// с непустым хостом.
        Админ-контролируемые фиды — риск низкий, но консервативный
        фикс не даёт случайным фидам открыть file:// / ftp:// и т.п.

        ⚠️  Остаточный риск: follow_redirects=True разрешает редиректы,
            валидный http(s) фид может ответить 30x → private IP / loopback.
            Фиды админ-контролируемые, риск минимальный. При необходимости
            отключить редиректы: httpx.Client(follow_redirects=False).
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return False
            # hostname непустой — отсекает "http://", "https://:8080", "https:///"
            if not parsed.hostname:
                return False
            return True
        except Exception:
            return False

    def close(self):
        self._client.close()

    def collect(self) -> Iterator[Tuple[str, FeedResult]]:
        """
        Генератор: (имя_фида, FeedResult).
        Фид без ошибки (пустой список) ≠ failed.
        """
        for src in self.sources:
            name = src.get("name", "unknown")
            url = src.get("url", "")
            if not url:
                logger.warning("[%s] URL пустой, пропускаем", name)
                continue
            if not self._validate_url(url):
                logger.warning("[%s] URL недопустим (scheme/host): %s", name, url)
                continue

            # lang из конфига фида (Task 2: lang проводка)
            lang = src.get("lang") or None

            result = self._fetch_feed(name, url, lang=lang)
            yield name, result

    def _fetch_feed(self, name: str, url: str, lang: Optional[str] = None) -> FeedResult:
        """
        GET + feedparser.

        Возвращает FeedResult(ok=True, items=[...]) при успехе
        (в т.ч. пустой список если фид легитимно пуст).
        Возвращает FeedResult(ok=False, items=[]) при любой ошибке,
        в том числе при превышении FEED_DEADLINE_SEC на HTTP-фазе.
        """
        # ── M1: единый monotonic-deadline на ВСЮ обработку фида ──
        # Выставляется ДО self._client.stream(): read-таймаут httpx
        # перезапускается на каждом чанке, поэтому медленный сервер
        # (по байту раз в 10 с) без этой проверки держал бы iter_bytes()
        # сколь угодно долго и останавливал весь цикл сбора.
        # monotonic вместо time.time(): системные часы могут скакнуть (NTP).
        feed_start = time.monotonic()
        deadline = feed_start + FEED_DEADLINE_SEC

        # ── HTTP: лимит по РАСПАКОВАННОМУ телу — защита от gzip-бомбы / гигантского фида ──
        try:
            logger.debug("[%s] GET %s", name, url)

            # Быстрый отказ по Content-Length (РАЗМЕР СЖАТОГО тела, не точный).
            # Не заменяет счётчик распакованных байтов — тот ниже.
            cl_str = ""
            with self._client.stream("GET", url) as resp:
                resp.raise_for_status()
                cl_str = resp.headers.get("content-length", "")
                try:
                    cl = int(cl_str)
                    if cl > MAX_FEED_BYTES * 3:
                        # Даже сжимаемый фид в 30 MB — подозрительно, отказ без итераций.
                        logger.warning("[%s] Content-Length %d >> limit %d, пропускаем",
                                       name, cl, MAX_FEED_BYTES)
                        return FeedResult(ok=False, items=[])
                except (ValueError, TypeError):
                    pass  # Content-Length нет или невалиден

                # Читаем поток, считая РАСПАКОВАННЫЕ байты (httpx декодирует автоматически).
                # resp.iter_bytes() — декодированные куски; bytearray растёт по распакованному размеру.
                buf = bytearray()
                for chunk in resp.iter_bytes(chunk_size=65536):
                    buf.extend(chunk)
                    if len(buf) > MAX_FEED_BYTES:
                        logger.warning("[%s] Feed too large (%d bytes): %s",
                                       name, len(buf), url)
                        return FeedResult(ok=False, items=[])
                    # M1: проверка deadline ВНУТРИ чтения — обрывает
                    # медленный сервер (иначе iter_bytes() бесконечен).
                    if time.monotonic() > deadline:
                        logger.warning(
                            "[%s] FEED_DEADLINE_SEC=%d превышен при чтении тела "
                            "(%d байт получено) — фид пропущен: %s",
                            name, FEED_DEADLINE_SEC, len(buf), url,
                        )
                        return FeedResult(ok=False, items=[])
                # M1: страховка после stream (медленное закрытие/финализация).
                if time.monotonic() > deadline:
                    logger.warning(
                        "[%s] FEED_DEADLINE_SEC=%d превышен после чтения тела "
                        "(%d байт) — фид пропущен: %s",
                        name, FEED_DEADLINE_SEC, len(buf), url,
                    )
                    return FeedResult(ok=False, items=[])
                raw = bytes(buf)
        except httpx.TimeoutException:
            logger.warning("[%s] Таймаут (%d с)", name, FETCH_TIMEOUT)
            return FeedResult(ok=False, items=[])
        except httpx.HTTPStatusError as e:
            logger.warning("[%s] HTTP %d: %s", name, e.response.status_code, url)
            return FeedResult(ok=False, items=[])
        except httpx.RequestError as e:
            logger.warning("[%s] Сеть недоступна: %s", name, e)
            return FeedResult(ok=False, items=[])
        except httpx.HTTPError as e:
            # TooManyRedirects, InvalidURL, UnsupportedProtocol и др.
            # наследуют HTTPError, но НЕ RequestError — ловим базовый.
            logger.warning("[%s] httpx.HTTPError: %s", name, e)
            return FeedResult(ok=False, items=[])
        except Exception as e:
            # Любая неожиданная ошибка → фид пропускается, цикл продолжается.
            logger.warning("[%s] Неожиданная ошибка сбора фида: %s", name, e)
            return FeedResult(ok=False, items=[])

        # ── Парсинг: raw (bytes) — автодекодировка XML по XML Declaration ──
        try:
            parsed = feedparser.parse(raw)
        except Exception as e:
            logger.warning("[%s] feedparser ошибка парсинга: %s", name, e)
            return FeedResult(ok=False, items=[])

        if parsed.bozo and not parsed.entries:
            logger.warning("[%s] Фид битый (bozo), entries пуст", name)
            return FeedResult(ok=False, items=[])

        # ── entries → NewsItem ──
        news_items = []
        # feed_start/deadline посчитаны выше (monotonic) — бюджет общий
        # для HTTP-фазы и конвертации entries.
        for entry in parsed.entries:
            # Wall-clock deadline: проверяем между entries (не убивая httpx-запрос)
            elapsed = time.monotonic() - feed_start
            if elapsed > FEED_DEADLINE_SEC:
                logger.warning(
                    "[%s] FEED_DEADLINE_SEC=%d превышен (%.1f с), "
                    "прерываем обработку фида, собрано %d item(s)",
                    name, FEED_DEADLINE_SEC, elapsed, len(news_items),
                )
                break

            # MAX_ITEMS_PER_FEED: защита от кривого фида
            if len(news_items) >= MAX_ITEMS_PER_FEED:
                logger.warning(
                    "[%s] MAX_ITEMS_PER_FEED=%d reached, "
                    "пропускаем остальные %d entries",
                    name, MAX_ITEMS_PER_FEED,
                    max(0, len(parsed.entries) - len(news_items)),
                )
                break

            link = getattr(entry, "link", "") or ""
            # пропускаем entry без URL (иначе url="" дедупится как один и тот же)
            if not link:
                logger.debug("[%s] пропущен entry без link: %s", name,
                             getattr(entry, "title", "")[:50])
                continue

            title = getattr(entry, "title", "") or ""
            # m3: feedparser отдаёт title с HTML-сущностями (AT&amp;T, &quot;).
            # Разэкранируем сразу — title идёт в дедуп, cluster_key и в БД,
            # а digest затем сам html.escape'ит при отправке. Без unescape
            # пользователь видел двойное экранирование (AT&amp;amp;T).
            title = _html.unescape(title)

            # summary: основной → fallback на content[0].value (<20 символов)
            summary_raw = getattr(entry, "summary", "") or ""
            if len(summary_raw) < 20:
                content = getattr(entry, "content", []) or []
                if content and hasattr(content[0], "value"):
                    summary_raw = content[0].value or ""
            summary = _strip_html(summary_raw)

            published_str = _format_pubdate(entry)
            pub_parsed = _parse_pubdate_utc(entry)

            # Читаем <category> теги из фида
            raw_tags = getattr(entry, "tags", []) or []
            tags_list = []
            for t in raw_tags:
                term = getattr(t, "term", "") or ""
                if term:
                    tags_list.append(term)
            tags_str = json.dumps(tags_list) if tags_list else None

            # cluster_key: локация|stem_hash|день публикации
            # Minor 6-фикс: единая реализация — build_cluster_key из locations.py
            cluster_key = None
            if title:
                from .locations import build_cluster_key, extract_location
                from .stemmer import stem_hash
                location = extract_location(title)
                stem = stem_hash(title)
                cluster_key = build_cluster_key(location, stem, published_str, title=title)

            news_items.append(NewsItem(
                source=name,
                url=link,
                title=title,
                summary=summary,
                published=published_str,
                published_parsed=pub_parsed,
                tags=tags_str,
                cluster_key=cluster_key,
                lang=lang,
            ))

        logger.info(
            "[%s] собрано %d item(s) из %s",
            name, len(news_items), url,
        )
        return FeedResult(ok=True, items=news_items)


# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────

# B6: максимальная длина summary для защиты LLM-бюджета и БД.
# Обрезается В collector (до отправки в LLM), дублируется в storage.
MAX_SUMMARY = 6000

# Collector Minor: тег = `<` + опциональный `/` + буква. Так «a < b > c»
# (сравнение/формула из текста) не считается тегом и не вырезается.
_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")


def _strip_html(text: str, max_len: int = MAX_SUMMARY) -> str:
    """
    Удаляем HTML-теги из summary, обрезаем до max_len символов (B6).

    m3: порядок преобразований — strip tags → unescape → повторный strip tags
    → нормализация пробелов. feedparser отдаёт текст с HTML-сущностями
    (&amp;, &quot;, &lt;) уже как есть; digest затем html.escape'ит при отправке,
    поэтому без unescape здесь пользователь видел двойное экранирование
    (AT&amp;amp;T в Telegram).

    Collector Minor: unescape ДО strip-tags ломал текст с формулами —
    "a &lt; b &gt; c" превращался в "a < b > c" и regex `<[^>]+>` вырезал
    "< b >". Поэтому: сначала снимаем настоящие теги, затем разэкранируем и
    снимаем теги ещё раз (сущность могла стать тегом: `&lt;b&gt;` → `<b>`,
    псевдотег вырезается). Tag-regex требует букву после `<`/`</` —
    "a < b > c" (пробел после `<`) остаётся текстом.
    """
    clean = _TAG_RE.sub(" ", text)
    clean = _html.unescape(clean)
    clean = _TAG_RE.sub(" ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:max_len]


def _parse_pubdate_utc(entry) -> Optional[datetime]:
    """
    Парсит published_parsed (struct_time) в UTC.
    Fallback: updated_parsed.
    Используем calendar.timegm вместо time.mktime,
    чтобы не было сдвига на локальный часовой пояс.
    """
    pt = getattr(entry, "published_parsed", None)
    if not pt:
        pt = getattr(entry, "updated_parsed", None)
    if not pt:
        return None
    try:
        return datetime.fromtimestamp(calendar.timegm(pt), tz=timezone.utc)
    except Exception:
        return None


def _format_pubdate(entry) -> str:
    """
    Формируем читаемую строку даты из entry.
    Приоритет: published_parsed (UTC) → updated_parsed → published → updated → '—'

    ВАЖНО: НЕ вызывает _parse_pubdate_utc — чтобы избежать двойного парсинга.
    Если нужна дата как datetime — вызывайте _parse_pubdate_utc() напрямую.
    """
    # Приоритет 1: published_parsed
    pt = getattr(entry, "published_parsed", None)
    if pt:
        try:
            dt = datetime.fromtimestamp(calendar.timegm(pt), tz=timezone.utc)
            return dt.strftime("%d.%m.%Y %H:%M UTC")
        except Exception:
            pass

    # Приоритет 2: updated_parsed (feedparser.struct_time → UTC → строка)
    ut = getattr(entry, "updated_parsed", None)
    if ut:
        try:
            dt = datetime.fromtimestamp(calendar.timegm(ut), tz=timezone.utc)
            return dt.strftime("%d.%m.%Y %H:%M UTC")
        except Exception:
            pass

    # фоллбэк на сырую строку
    for attr in ("published", "updated"):
        raw = getattr(entry, attr, None)
        if raw:
            return str(raw)
    return "—"