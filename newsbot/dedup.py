"""
Дедупликация: обёртка над Storage.

NOTE: дедуп по точному URL (primary key); URL-нормализация (utm/trailing slash)
для future — не применяется в текущей реализации.
"""
import logging
from typing import List, Tuple, Set

from .collector import NewsItem
from .storage import Storage, MIN_TITLE_LEN

logger = logging.getLogger("newsbot.dedup")


def _is_meaningful_title(item: NewsItem) -> bool:
    """Заголовок достаточно осмысленный для title-based дедупа (>= MIN_TITLE_LEN)."""
    return len(item.title_normalized()) >= MIN_TITLE_LEN


def filter_new(items: List[NewsItem], storage: Storage) -> Tuple[List[NewsItem], int, int]:
    """
    Фильтрует список items, оставляя только новые (не дубли).

    Двухуровневая проверка:
      (a) точный URL в БД — permanent dedup (URL не меняется)
      (b) нормализованный заголовок + source в пределах TTL
      (c) нормализованный заголовок + source ВНУТРИ текущего цикла
          (внутрицикловой dedup: два разных URL с одинаковым title+source)

    Возвращает (new_items, total_count, dup_count).
    """
    new_items = []
    dup_count = 0

    # Множество ключей (normalized_title, source), встреченных в этом цикле.
    # Ключ — None для пустых/коротких заголовков (не участвуют в title-dedup).
    seen_in_cycle: Set[tuple] = set()

    for item in items:
        # (a) точный URL в БД
        if storage.is_duplicate(item):
            dup_count += 1
            logger.debug("URL-дубликат пропущен: %s", item.url[:80])
            continue

        normalized = item.title_normalized()

        # (b) title+source в БД (проверяется внутри is_duplicate → уже прошли)
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

        new_items.append(item)

    return new_items, len(items), dup_count