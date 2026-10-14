"""
Коллектор RSS-фидов: feedparser + httpx.
Один фид не роняет цикл.

Корректность:
  - resp.content (bytes) → feedparser.parse() — автодекодинг по XML Declaration
  - published_parsed → calendar.timegm (UTC) вместо time.mktime (localtime)
  - пустые entry без link пропускаются
"""
import calendar
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, List, Optional, Tuple

import feedparser
import httpx

logger = logging.getLogger("newsbot.collector")

FETCH_TIMEOUT = 15  # секунд
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
    published_parsed: Optional[datetime]
    # --- результат LLM-фильтра (этап 2) ---
    llm_score: Optional[int] = None          # оценка 0..10
    llm_summary: Optional[str] = None         # выжимка от LLM
    llm_reason: Optional[str] = None          # краткое пояснение
    retry_count: int = 0                      # счётчик попыток retry
    fetched_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # --- Фаза 2b: transient runtime-поле для кнопок фидбэка ---
    # Не является колонкой БД; заполняется в pipeline перед отправкой push-сообщения.
    feedback_target_id: Optional[int] = None

    def title_normalized(self) -> str:
        """Нормализованный заголовок для дедупа: lowercase, без пунктуации."""
        return re.sub(r"[^\w\s]", "", self.title.lower())


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

            result = self._fetch_feed(name, url)
            yield name, result

    def _fetch_feed(self, name: str, url: str) -> FeedResult:
        """
        GET + feedparser.

        Возвращает FeedResult(ok=True, items=[...]) при успехе
        (в т.ч. пустой список если фид легитимно пуст).
        Возвращает FeedResult(ok=False, items=[]) при любой ошибке.
        """
        # ── HTTP ──
        try:
            logger.debug("[%s] GET %s", name, url)
            resp = self._client.get(url)
            resp.raise_for_status()
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

        # ── Парсинг: resp.content (bytes) — корректная автодекодировка XML ──
        try:
            parsed = feedparser.parse(resp.content)
        except Exception as e:
            logger.warning("[%s] feedparser ошибка парсинга: %s", name, e)
            return FeedResult(ok=False, items=[])

        if parsed.bozo and not parsed.entries:
            logger.warning("[%s] Фид битый (bozo), entries пуст", name)
            return FeedResult(ok=False, items=[])

        # ── entries → NewsItem ──
        news_items = []
        for entry in parsed.entries:
            link = getattr(entry, "link", "") or ""
            # пропускаем entry без URL (иначе url="" дедупится как один и тот же)
            if not link:
                logger.debug("[%s] пропущен entry без link: %s", name,
                             getattr(entry, "title", "")[:50])
                continue

            title = getattr(entry, "title", "") or ""
            summary = _strip_html(getattr(entry, "summary", "") or "")
            published_str = _format_pubdate(entry)
            pub_parsed = _parse_pubdate_utc(entry)

            news_items.append(NewsItem(
                source=name,
                url=link,
                title=title,
                summary=summary,
                published=published_str,
                published_parsed=pub_parsed,
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


def _strip_html(text: str, max_len: int = MAX_SUMMARY) -> str:
    """Удаляем HTML-теги из summary, обрезаем до max_len символов (B6)."""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:max_len]


def _parse_pubdate_utc(entry) -> Optional[datetime]:
    """
    Парсит published_parsed (struct_time) в UTC.
    Используем calendar.timegm вместо time.mktime,
    чтобы не было сдвига на локальный часовой пояс.
    """
    pt = getattr(entry, "published_parsed", None)
    if not pt:
        return None
    try:
        return datetime.fromtimestamp(calendar.timegm(pt), tz=timezone.utc)
    except Exception:
        return None


def _format_pubdate(entry) -> str:
    """
    Формируем читаемую строку даты из entry.
    Приоритет: published_parsed (UTC) → published → updated → '—'

    ВАЖНО: НЕ вызывает _parse_pubdate_utc — чтобы избежать двойного парсинга.
    Если нужна дата как datetime — вызывайте _parse_pubdate_utc() напрямую.
    """
    pt = getattr(entry, "published_parsed", None)
    if pt:
        try:
            dt = datetime.fromtimestamp(calendar.timegm(pt), tz=timezone.utc)
            return dt.strftime("%d.%m.%Y %H:%M UTC")
        except Exception:
            pass

    # фоллбэк на сырую строку
    for attr in ("published", "updated"):
        raw = getattr(entry, attr, None)
        if raw:
            return str(raw)
    return "—"