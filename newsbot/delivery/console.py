# -*- coding: utf-8 -*-
"""
ConsoleDelivery - output news to stdout.
"""
import logging
from typing import List, Optional, Protocol, Tuple

from ..storage import NewsItem
from . import format_time_utc

logger = logging.getLogger("newsbot.delivery.console")


class Delivery(Protocol):
    """
    Base delivery protocol.
    Both implementations return Tuple[List[str], List[str]] -
    (successful_urls, permanent_failed_urls). Console has no permanent failures.

    Contract: send_text / send_digest receive text that is ALREADY valid HTML
    (already escaped if necessary). Implementations MUST NOT re-escape it.
    """

    def send(self, items: List[NewsItem]) -> Tuple[List[str], List[str]]:
        """Single send — alias for send_batch. Returns (successful_urls, permanent_failed_urls)."""
        ...

    def send_batch(self, items: List[NewsItem]) -> Tuple[List[str], List[str]]:
        """Return Tuple[List[str], List[str]]: (successful_urls, permanent_failed_urls)."""
        ...

    def send_text(self, text: str) -> bool:
        """
        Send arbitrary HTML text to owner. Return True on success.
        CONTRACT: text is already valid HTML — do NOT re-escape.
        """
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
    def send(self, items: List[NewsItem]) -> Tuple[List[str], List[str]]:
        """Single send — alias for send_batch."""
        return self.send_batch(items)

    def send_batch(self, items: List[NewsItem]) -> Tuple[List[str], List[str]]:
        """
        Print items to stdout.
        Return (urls, permanent_failed_urls).
        Console has no permanent failures — second element is always [].
        """
        if not items:
            return [], []

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

        return [item.url for item in items], []

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