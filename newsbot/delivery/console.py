# -*- coding: utf-8 -*-
"""
ConsoleDelivery - output news to stdout.
"""
import logging
from typing import List, Optional, Protocol

from ..storage import NewsItem
from . import format_time_utc

logger = logging.getLogger("newsbot.delivery.console")


class Delivery(Protocol):
    """
    Base delivery protocol.
    Both implementations return List[str] - URLs of successfully delivered items.
    """

    def send_batch(self, items: List[NewsItem]) -> List[str]:
        """Return List[str]: URLs of successfully delivered items."""
        ...

    def send_text(self, text: str) -> bool:
        """Send arbitrary text to owner. Return True on success."""
        ...

    def send_digest(self, text: str, item_buttons: List[dict]) -> bool:
        """
        Send digest with per-item inline buttons.
        item_buttons: List[dict] with {url, target_id} per item.
        Console: ignores item_buttons, prints text.
        Returns True.
        """
        ...

    def close(self) -> None:
        ...


class ConsoleDelivery(Delivery):
    def send(self, items: List[NewsItem]) -> List[str]:
        """Single send - alias for send_batch."""
        return self.send_batch(items)

    def send_batch(self, items: List[NewsItem]) -> List[str]:
        """
        Print items to stdout.
        Return list of URLs of printed items.
        Contract matches TelegramDelivery: pipeline expects List[str].
        """
        if not items:
            return []

        for item in items:
            body = item.llm_summary if item.llm_summary else item.summary
            print(f"\n{'='*60}")
            print(f"[{item.source}] {item.title}")
            if body:
                print(f"   {body[:300]}")
            if item.llm_score is not None:
                print(f"   [LLM score={item.llm_score}]")
            # N1/N6: published_parsed если есть, иначе fetched_at; local timezone
            pub_parsed = getattr(item, "published_parsed", None)
            time_str = format_time_utc(pub_parsed, item.fetched_at)
            marker = " (published)" if pub_parsed else " (fetched)"
            print(f"   [{time_str}]{marker}")
            if item.url:
                print(f"   >> {item.url}")
        print(f"\n{'='*60}  Всего: {len(items)} новостей")
        logger.info("ConsoleDelivery: выведено %d новостей", len(items))

        return [item.url for item in items]

    def send_text(self, text: str) -> bool:
        """Print text to stdout."""
        print(text)
        return True

    def send_digest(self, text: str, item_buttons: List[dict]) -> bool:
        """
        Send digest message (ignores item_buttons — Console has no buttons).
        item_buttons: List[dict] with {url, target_id} per item.
        Returns True always (Console is always "successful").
        """
        return self.send_text(text)

    def close(self) -> None:
        pass