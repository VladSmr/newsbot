"""
test_collector_hardening.py — ремонт по вердикту критика (collector.py).

M1 — wall-clock (monotonic) deadline покрывает и HTTP-фазу: медленный сервер,
     отдающий по чанку, не держит iter_bytes() дольше FEED_DEADLINE_SEC;
     фид помечается ok=False, цикл продолжается со следующим фидом.
m3 — HTML-сущности разэкранируются (html.unescape): title и summary
     не приходят в digest двойноэкрaнированными (AT&amp;T → AT&T).
"""
import time

import pytest

from newsbot.collector import (
    FEED_DEADLINE_SEC,
    FeedCollector,
    _strip_html,
)


_RSS_FAST = b'''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Fast Feed</title>
  <link>http://fast.example</link>
  <item>
    <title>AT&amp;amp;T buys &amp;quot;Thing&amp;quot;</title>
    <link>http://fast.example/news/1</link>
    <description>&lt;p&gt;AT&amp;amp;T and &amp;amp; more&lt;/p&gt;</description>
    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
  </item>
</channel>
</rss>'''


class _FakeResponse:
    """Минимальный двойник httpx.Response: только то, что нужно _fetch_feed."""

    def __init__(self, chunks, delay=0.0, headers=None):
        self._chunks = chunks
        self._delay = delay
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def iter_bytes(self, chunk_size=65536):
        for chunk in self._chunks:
            if self._delay:
                time.sleep(self._delay)
            yield chunk


class _FakeStream:
    """Контекстный менеджер-двойник httpx.Client.stream(...)."""

    def __init__(self, response, enter_delay=0.0):
        self._response = response
        self._enter_delay = enter_delay

    def __enter__(self):
        if self._enter_delay:
            time.sleep(self._enter_delay)
        return self._response

    def __exit__(self, *args):
        return False


@pytest.fixture
def collector_factory(monkeypatch):
    """Фабрика FeedCollector с подменённым self._client.stream."""
    created = []

    def _make(sources, stream_factory):
        collector = FeedCollector(sources)
        monkeypatch.setattr(collector._client, "stream", stream_factory)
        created.append(collector)
        return collector

    yield _make

    for c in created:
        c.close()


class TestStreamDeadline:
    """M1: deadline действует на чтение тела, а не только на конвертацию."""

    def test_deadline_defined(self):
        # NIT: не пинним конкретное значение — поведение проверяют тесты
        # ниже (deadline реально обрывает медленный стрим).
        assert isinstance(FEED_DEADLINE_SEC, (int, float))
        assert FEED_DEADLINE_SEC > 0

    def test_slow_stream_aborts_with_ok_false(self, monkeypatch, collector_factory):
        """
        Сервер отдаёт по чанку раз в 50 мс → deadline срабатывает ВНУТРИ
        iter_bytes(); фид возвращается как ok=False, items=[].
        """
        monkeypatch.setattr(
            "newsbot.collector.FEED_DEADLINE_SEC", 0.3,
        )
        chunks = [b"<rss>" + b"x" * 10] * 200  # ~10 с при 50 мс/чанк

        def stream_factory(method, url, **kwargs):
            return _FakeStream(_FakeResponse(chunks, delay=0.05))

        collector = collector_factory(
            [{"name": "slow", "url": "http://slow.example/feed"}],
            stream_factory,
        )

        started = time.monotonic()
        name, result = next(collector.collect())
        elapsed = time.monotonic() - started

        assert name == "slow"
        assert result.ok is False, "медленный стрим должен обрываться по deadline"
        assert result.items == []
        assert elapsed < 5, f"deadline не сработал вовремя: {elapsed:.1f} с"

    def test_cycle_continues_after_deadline(self, monkeypatch, collector_factory):
        """
        После обрыва первого фида цикл продолжается: следующий фид собирается.
        """
        monkeypatch.setattr(
            "newsbot.collector.FEED_DEADLINE_SEC", 0.3,
        )

        def stream_factory(method, url, **kwargs):
            if "slow" in url:
                return _FakeStream(_FakeResponse([b"x" * 10] * 200, delay=0.05))
            return _FakeStream(_FakeResponse([_RSS_FAST]))

        collector = collector_factory(
            [
                {"name": "slow", "url": "http://slow.example/feed"},
                {"name": "fast", "url": "http://fast.example/feed"},
            ],
            stream_factory,
        )

        results = list(collector.collect())

        assert [r[0] for r in results] == ["slow", "fast"]
        assert results[0][1].ok is False
        assert results[1][1].ok is True
        assert len(results[1][1].items) == 1

    def test_fast_stream_not_affected(self, collector_factory):
        """Быстрый фид не обрывается deadline'ом."""
        collector = collector_factory(
            [{"name": "fast", "url": "http://fast.example/feed"}],
            lambda method, url, **kwargs: _FakeStream(_FakeResponse([_RSS_FAST])),
        )
        _, result = next(collector.collect())
        assert result.ok is True
        assert len(result.items) == 1


class TestHtmlUnescape:
    """m3: html.unescape перед strip-tags/нормализацией."""

    def test_strip_html_unescapes_entities(self):
        assert _strip_html("<p>AT&amp;T and &quot;quote&quot;</p>") == 'AT&T and "quote"'

    def test_strip_html_entities_before_tags(self):
        """&lt;b&gt; превращается в тег уже после unescape и вырезается."""
        assert _strip_html("&lt;b&gt;bold&lt;/b&gt; text") == "bold text"

    def test_strip_html_preserves_comparison_text(self):
        """
        Collector Minor: unescape до strip-tags терял текст с формулами —
        "a &lt; b &gt; c" → "a < b > c" → regex <[^>]+> вырезал "< b >".
        Теперь теги снимаются до unescape, а tag-regex требует букву после
        '<', поэтому сравнение сохраняется.
        """
        assert _strip_html("a &lt; b &gt; c") == "a < b > c"

    def test_strip_html_removes_escaped_pseudotag(self):
        """Псевдотег из сущностей (&lt;b&gt;) всё равно вырезается."""
        assert _strip_html("x &lt;b&gt; y") == "x y"

    def test_strip_html_max_len(self):
        assert len(_strip_html("a" * 50, max_len=10)) == 10

    def test_feed_title_and_summary_unescaped(self, collector_factory):
        """Из фида: title и summary приходят без HTML-сущностей."""
        collector = collector_factory(
            [{"name": "fast", "url": "http://fast.example/feed"}],
            lambda method, url, **kwargs: _FakeStream(_FakeResponse([_RSS_FAST])),
        )
        _, result = next(collector.collect())
        item = result.items[0]

        assert item.title == 'AT&T buys "Thing"', item.title
        assert "&amp;" not in item.summary, item.summary
        assert "AT&T and & more" == item.summary
