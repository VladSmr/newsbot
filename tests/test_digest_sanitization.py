"""
Task 2: тесты — control-chars в digest-рендере не попадают в итоговый текст.

Запускать: python -m pytest tests/test_digest_sanitization.py -v
"""
import pytest

from newsbot.digest import build_digest_messages, _render_item_raw
from newsbot.storage import NewsItem


def _make_item(
    title: str = "Normal Title",
    source: str = "Normal Source",
    summary: str = "Normal summary text.",
    url: str = "https://example.com/news/1",
) -> NewsItem:
    """Factory для NewsItem с минимумом обязательных полей."""
    from datetime import datetime, timezone
    return NewsItem(
        source=source,
        url=url,
        title=title,
        summary=summary,
        published="01.01.2024 12:00 UTC",
        fetched_at=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
    )


class TestRenderItemRawControlChars:
    """_render_item_raw удаляет control-chars из title/source/body."""

    def test_null_char_removed_from_title(self):
        """\\x00 удаляется из title."""
        item = _make_item(title="Breaking\x00News: Test")
        title, *_ = _render_item_raw(item)
        assert "\x00" not in title
        assert "BreakingNews: Test" in title

    def test_vertical_tab_removed_from_title(self):
        """\\x0b удаляется из title."""
        item = _make_item(title="News\x0bUpdate")
        title, *_ = _render_item_raw(item)
        assert "\x0b" not in title
        assert "NewsUpdate" in title

    def test_escape_char_removed_from_title(self):
        """\\x1b удаляется из title."""
        item = _make_item(title="Alert\x1bSystem")
        title, *_ = _render_item_raw(item)
        assert "\x1b" not in title
        assert "AlertSystem" in title

    def test_line_separator_removed_from_title(self):
        """\\u2028 удаляется из title."""
        item = _make_item(title="News\u2028Update")
        title, *_ = _render_item_raw(item)
        assert "\u2028" not in title

    def test_paragraph_separator_removed_from_title(self):
        """\\u2029 удаляется из title."""
        item = _make_item(title="News\u2029Text")
        title, *_ = _render_item_raw(item)
        assert "\u2029" not in title

    def test_control_chars_removed_from_source(self):
        """\\x00, \\x0b, \\x1b, \\u2028 удаляются из source."""
        item = _make_item(source="Source\x00Name\x0bInc\x1b\u2028")
        _, source, _, _ = _render_item_raw(item)
        assert "\x00" not in source
        assert "\x0b" not in source
        assert "\x1b" not in source
        assert "\u2028" not in source
        assert "SourceNameInc" in source

    def test_control_chars_removed_from_summary(self):
        """\\x00, \\x0b, \\x1b, \\u2028 удаляются из body (summary)."""
        item = _make_item(
            summary="This is a\x00test.\x0bNew para.\x1bEnd.\u2028Final."
        )
        _, _, body, _ = _render_item_raw(item)
        assert "\x00" not in body
        assert "\x0b" not in body
        assert "\x1b" not in body
        assert "\u2028" not in body
        # Контрольные символы удалены — слова без разделителей сливаются
        # (это ожидаемо: мы удаляем, не заменяем на пробел).
        assert "This is atest." in body  # \x00 убран
        assert "Final." in body         # \u2028 убран

    def test_full_block_of_forbidden_chars(self):
        """Все символы из паттерна [\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f\\x85\\u2028\\u2029] удалены."""
        # Собираем строку со всеми forbidden-символами
        forbidden = "".join(chr(c) for c in (
            list(range(0, 9))       # \x00-\x08
            + [0x0b, 0x0c]          # \x0b, \x0c
            + list(range(0x0e, 0x20))  # \x0e-\x1f
            + [0x7f, 0x85]          # \x7f, \x85
            + [0x2028, 0x2029]      # \u2028, \u2029
        ))
        item = _make_item(
            title=f"Title{forbidden}",
            source=f"Source{forbidden}",
            summary=f"Summary{forbidden}",
        )
        title, source, body, _ = _render_item_raw(item)
        for r in (title, source, body):
            for ch in forbidden:
                assert ch not in r, f"Control char {repr(ch)} found in {repr(r)}"

    def test_normal_text_preserved(self):
        """Обычный текст (без control-chars) не искажается."""
        item = _make_item(
            title="Breaking News: Major Event Today",
            source="Reuters",
            summary="This is a normal summary with numbers 123 and symbols !?@.",
        )
        title, source, body, _ = _render_item_raw(item)
        assert title == "Breaking News: Major Event Today"
        assert source == "Reuters"
        assert "This is a normal summary" in body

    def test_newlines_tabs_carriage_return_preserved(self):
        """\\n, \\t, \\r сохраняются (имеют смысл в тексте)."""
        item = _make_item(
            title="Title\nwith\nnewlines",
            source="Source\twith\ttabs",
            summary="Line1\r\nLine2\rLine3",
        )
        title, source, body, _ = _render_item_raw(item)
        assert "\n" in title
        assert "\t" in source
        assert "\r" in body

    def test_llm_title_overrides_title(self):
        """llm_title приоритетнее title, control-chars тоже удаляются."""
        item = _make_item(
            title="Title\x00Bad",
            summary="Summary",
        )
        item.llm_title = "LLM Title\x1bIs Clean"
        title, *_ = _render_item_raw(item)
        assert "\x1b" not in title
        assert "LLM TitleIs Clean" in title

    def test_html_entities_not_escaped_in_raw(self):
        """_render_item_raw возвращает сырые строки, НЕ html.escape'нутые."""
        item = _make_item(
            title="Title with <special> &amp; chars",
            source="Source",
            summary="Summary <b>bold</b> text.",
        )
        title, source, body, _ = _render_item_raw(item)
        # В raw-версии сущности НЕ экранированы — html.escape делает
        # вызывающая сторона (build_digest_messages). Control-chars уже убраны.
        assert "<special>" in title  # не &lt;special&gt;
        assert "<b>bold</b>" in body


class TestBuildDigestMessagesControlChars:
    """build_digest_messages не пропускает control-chars в итоговый HTML."""

    def test_control_chars_in_multiple_items(self):
        """Items с control-chars в title/summary/source рендерятся без них."""
        items = [
            _make_item(
                title=f"News Item {i}\x00\x1b\u2028",
                source=f"Source {i}\x0b",
                summary=f"Summary with control\x07char {i}.",
                url=f"https://example.com/{i}",
            )
            for i in range(5)
        ]
        groups, all_included, failed_urls = build_digest_messages(items)

        assert len(groups) >= 1
        # Все группы (HTML-тексты) не содержат control-chars
        forbidden = "".join(chr(c) for c in (
            list(range(0, 9))
            + [0x0b, 0x0c]
            + list(range(0x0e, 0x20))
            + [0x7f, 0x85]
            + [0x2028, 0x2029]
        ))
        for msg, _ in groups:
            for ch in forbidden:
                assert ch not in msg, (
                    f"Control char {repr(ch)} found in digest message"
                )

    def test_normal_items_unchanged(self):
        """Обычные items рендерятся корректно (регрессия)."""
        items = [
            _make_item(
                title="First Important News",
                source="Reuters",
                summary="Detailed summary of the first news item.",
            ),
            _make_item(
                title="Second Update",
                source="BBC",
                summary="Another important update for today.",
            ),
        ]
        groups, all_included, failed_urls = build_digest_messages(items)

        assert len(groups) == 1
        msg, _ = groups[0]
        assert "First Important News" in msg
        assert "Second Update" in msg
        assert len(all_included) == 2
        assert failed_urls == []

    def test_mixed_normal_and_control_char_items(self):
        """Смесь нормальных items и items с control-chars — все корректны."""
        normal = _make_item(
            title="Normal News Title",
            source="Normal Source",
            summary="Normal summary text.",
        )
        bad_title = _make_item(
            title="Bad\x00Title\x1b",
            source="Source",
            summary="Summary",
        )
        bad_summary = _make_item(
            title="Title",
            source="Source",
            summary="Summary\x0bwith\x07bad\u2028chars",
        )
        another_normal = _make_item(
            title="Another Normal Title",
            source="Another Source",
            summary="Another summary.",
        )
        items = [normal, bad_title, bad_summary, another_normal]
        groups, all_included, failed_urls = build_digest_messages(items)

        # Все 4 items должны быть включены (control-chars удалены, не ошибка)
        assert len(all_included) == 4, (
            f"Expected 4 items, got {len(all_included)}. "
            f"failed_urls={failed_urls}"
        )
        assert failed_urls == []
        # Итоговые сообщения не содержат control-chars
        forbidden = "".join(chr(c) for c in (
            list(range(0, 9))
            + [0x0b, 0x0c]
            + list(range(0x0e, 0x20))
            + [0x7f, 0x85]
            + [0x2028, 0x2029]
        ))
        for msg, _ in groups:
            for ch in forbidden:
                assert ch not in msg