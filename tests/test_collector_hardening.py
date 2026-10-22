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
    CONNECT_TIMEOUT_CAP,
    FEED_DEADLINE_SEC,
    FeedCollector,
    MAX_REDIRECTS,
    MIN_PHASE_TIMEOUT,
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


# ─────────────────────────────────────────────────────────────────
# P1: FEED_DEADLINE_SEC не жёсткий — редиректы + остаток бюджета
# ─────────────────────────────────────────────────────────────────

class TestRequestBudget:
    """P1: каждая HTTP-операция получает ОСТАТОК бюджета фида."""

    def test_client_max_redirects(self):
        """max_redirects ограничен (иначе 20 хопов × таймаут держат цикл)."""
        from newsbot.collector import MAX_REDIRECTS

        collector = FeedCollector([])
        try:
            assert collector._client.max_redirects == MAX_REDIRECTS
            assert MAX_REDIRECTS <= 5
        finally:
            collector.close()

    def test_stream_gets_remaining_budget_timeout(self, collector_factory, monkeypatch):
        """
        Per-hop timeout = остаток бюджета / (MAX_REDIRECTS + 1).
        httpx применяет Timeout к КАЖДОМУ хопу цепочки редиректов, поэтому
        отдавать на запрос весь остаток нельзя (худший случай — 4×остаток).
        Проверяем при остатке 20 с: старый код дал бы 15 с (FETCH_TIMEOUT),
        новый — 5 с, и сумма по цепочке укладывается в остаток.
        """
        import newsbot.collector as collector_mod
        from newsbot.collector import FEED_DEADLINE_SEC, FETCH_TIMEOUT, MAX_REDIRECTS

        captured = {}

        def stream_factory(method, url, **kwargs):
            captured.update(kwargs)
            return _FakeStream(_FakeResponse([_RSS_FAST]))

        # Часы: feed_start, затем «осталось 20 с бюджета» (и так далее).
        fake_now = 1000.0
        remaining = 20.0
        calls = {"n": 0}

        def fake_monotonic():
            calls["n"] += 1
            return fake_now if calls["n"] == 1 else fake_now + (FEED_DEADLINE_SEC - remaining)

        monkeypatch.setattr(collector_mod.time, "monotonic", fake_monotonic)

        collector = collector_factory(
            [{"name": "fast", "url": "http://fast.example/feed"}],
            stream_factory,
        )
        next(collector.collect())

        timeout = captured.get("timeout")
        assert timeout is not None, "timeout должен передаваться на запрос"
        read_timeout = timeout.read if hasattr(timeout, "read") else timeout

        assert read_timeout < FETCH_TIMEOUT, (
            f"per-hop timeout должен быть меньше FETCH_TIMEOUT при остатке "
            f"{remaining} с, получено {read_timeout}"
        )
        assert read_timeout <= remaining / (MAX_REDIRECTS + 1) + 0.01, (
            f"per-hop timeout {read_timeout} не укладывается в остаток "
            f"{remaining} с на {MAX_REDIRECTS + 1} хопов"
        )
        # Сумма по всей цепочке редиректов не превышает остаток бюджета.
        assert read_timeout * (MAX_REDIRECTS + 1) <= remaining + 0.01, (
            f"{MAX_REDIRECTS + 1} хопов × {read_timeout} с превышают остаток {remaining} с"
        )


# ─────────────────────────────────────────────────────────────────
# Minor (волна 4): httpx-таймаут действует на КАЖДУЮ фазу → бюджет хопа
# делится на connect/read, иначе худший случай хопа = 2 × hop_timeout.
# ─────────────────────────────────────────────────────────────────

def _capture_timeout(collector_factory, monkeypatch, remaining,
                     stream_factory=None):
    """
    Прогоняет один фид с искусственным «остатком бюджета» remaining
    (подменяем monotonic) и возвращает httpx.Timeout, переданный
    в client.stream(). Первый вызов monotonic — feed_start, остальные —
    момент расчёта remaining (ровно deadline - remaining).
    """
    import newsbot.collector as collector_mod
    from newsbot.collector import FEED_DEADLINE_SEC

    sent = {}
    base = 1000.0
    budget = FEED_DEADLINE_SEC  # на случай monkeypatch в других тестах
    calls = {"n": 0}

    def fake_monotonic():
        calls["n"] += 1
        return base if calls["n"] == 1 else base + (budget - remaining)

    monkeypatch.setattr(collector_mod.time, "monotonic", fake_monotonic)

    def default_stream(method, url, **kwargs):
        return _FakeStream(_FakeResponse([_RSS_FAST]))

    def capturing_stream(method, url, **kwargs):
        sent.update(kwargs)
        return (stream_factory or default_stream)(method, url, **kwargs)

    collector = collector_factory(
        [{"name": "fast", "url": "http://fast.example/feed"}],
        capturing_stream,
    )
    next(collector.collect())
    return sent.get("timeout")


class TestPerHopPhaseSplit:
    """Minor: connect и read получают по половине бюджета хопа."""

    def test_connect_and_read_split_hop_budget(self, collector_factory, monkeypatch):
        """
        Остаток 20 с → hop_budget 5 с → phase_budget 2.5 с.
        Раньше Timeout(hop) давал 5 с на КАЖДУЮ фазу (худший случай хопа 10 с).
        Теперь connect + read ≤ hop_budget.
        """
        timeout = _capture_timeout(collector_factory, monkeypatch, remaining=20.0)

        assert timeout is not None, "timeout должен передаваться на запрос"
        hop_budget = 20.0 / (MAX_REDIRECTS + 1)

        assert timeout.connect == pytest.approx(hop_budget / 2), timeout
        assert timeout.read == pytest.approx(hop_budget / 2), timeout
        assert timeout.connect + timeout.read <= hop_budget + 1e-6, (
            f"connect {timeout.connect} + read {timeout.read} не укладываются "
            f"в бюджет хопа {hop_budget}"
        )
        assert timeout.connect <= CONNECT_TIMEOUT_CAP + 1e-6

    def test_worst_case_chain_fits_remaining_budget(self, collector_factory, monkeypatch):
        """
        Инвариант: (connect + read) × (MAX_REDIRECTS + 1) ≤ remaining.
        Полный бюджет фида (30 с) → 4 хопа × 7.5 с = 30 с, не 60 с.
        """
        remaining = 30.0
        timeout = _capture_timeout(collector_factory, monkeypatch, remaining=remaining)

        worst_per_hop = timeout.connect + timeout.read
        worst_chain = worst_per_hop * (MAX_REDIRECTS + 1)

        assert worst_chain <= remaining + 1e-6, (
            f"худший случай цепочки {worst_chain} с превышает остаток {remaining} с"
        )

    def test_floor_documented_for_tiny_budget(self, collector_factory, monkeypatch):
        """
        Floor MIN_PHASE_TIMEOUT=1.0 с: при почти исчерпанном бюджете обе фазы
        равны floor'у, худший случай цепочки ограничен
        (MAX_REDIRECTS + 1) × 2 × MIN_PHASE_TIMEOUT — это осознанный размен,
        зафиксирован в докстринге _fetch_feed (таймаут < 1 с рвёт быстрые соединения).
        """
        remaining = 0.2
        timeout = _capture_timeout(collector_factory, monkeypatch, remaining=remaining)

        assert timeout.connect == pytest.approx(MIN_PHASE_TIMEOUT), timeout
        assert timeout.read == pytest.approx(MIN_PHASE_TIMEOUT), timeout

        worst_chain = (timeout.connect + timeout.read) * (MAX_REDIRECTS + 1)
        ceiling = max(remaining, (MAX_REDIRECTS + 1) * 2 * MIN_PHASE_TIMEOUT)
        assert worst_chain <= ceiling + 1e-6, (
            f"худший случай {worst_chain} с выше документированного потолка {ceiling} с"
        )

    def test_timeout_log_reports_actual_phases(
        self, collector_factory, monkeypatch, caplog,
    ):
        """
        Nit: в лог таймаута попадают ФАКТИЧЕСКИЕ connect/read (2.5/2.5 при
        остатке 20 с), а не константа FETCH_TIMEOUT=15.
        """
        import logging

        import httpx

        def failing_stream(method, url, **kwargs):
            raise httpx.ReadTimeout("test timeout")

        caplog.set_level(logging.WARNING, logger="newsbot.collector")

        # Переиспользуем _capture_timeout: он подменяет monotonic и прогоняет
        # collect(); исключение ловится внутри _fetch_feed → ok=False.
        _capture_timeout(
            collector_factory, monkeypatch, remaining=20.0,
            stream_factory=failing_stream,
        )

        assert "connect 2.5" in caplog.text, caplog.text
        assert "read 2.5" in caplog.text, caplog.text
        assert "Таймаут (per-hop: connect" in caplog.text, caplog.text


# ─────────────────────────────────────────────────────────────────
# P2: ошибка конвертации одного entry не убивает остальные фиды
# ─────────────────────────────────────────────────────────────────

_RSS_BROKEN_ENTRY = b'''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
<channel>
  <title>Broken</title>
  <link>http://broken.example</link>
  <item>
    <title>Broken entry</title>
    <link>http://broken.example/1</link>
  </item>
</channel>
</rss>'''


class TestEntryIsolation:
    """P2: исключение на одном entry → warning + continue, фиды не теряются."""

    def test_entry_exception_does_not_kill_following_feeds(
        self, monkeypatch, collector_factory,
    ):
        import newsbot.collector as collector_mod

        real = collector_mod._entry_to_news_item
        calls = {"n": 0}

        def flaky(entry, name, link, lang):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("битый entry")
            return real(entry, name, link, lang)

        monkeypatch.setattr(collector_mod, "_entry_to_news_item", flaky)

        feeds = {
            "http://broken.example/feed": _RSS_BROKEN_ENTRY,
            "http://fast.example/feed": _RSS_FAST,
        }

        def stream_factory(method, url, **kwargs):
            return _FakeStream(_FakeResponse([feeds[url]]))

        collector = collector_factory(
            [
                {"name": "broken", "url": "http://broken.example/feed"},
                {"name": "fast", "url": "http://fast.example/feed"},
            ],
            stream_factory,
        )

        results = list(collector.collect())

        assert len(results) == 2, "второй фид должен обработаться после сбоя entry"
        assert results[0][0] == "broken"
        assert results[0][1].ok is True
        assert results[0][1].items == [], "битый entry пропущен"
        assert results[1][0] == "fast"
        assert len(results[1][1].items) == 1

    def test_non_str_tag_term_does_not_break_entry(self, collector_factory):
        """Не-str term в <category> не роняет конвертацию entry."""
        from newsbot.collector import _entry_to_news_item

        class _Tag:
            term = 123

        class _Entry:
            title = "Normal title"
            link = "http://x.example/1"
            summary = "summary text long enough"
            tags = [_Tag()]
            published = "Mon, 01 Jan 2024 12:00:00 GMT"

        item = _entry_to_news_item(_Entry(), "src", "http://x.example/1", "en")
        assert item is not None
        assert item.tags is not None
        assert "123" in item.tags


class TestTitleCap:
    """P2: title капится и в collector (до БД/LLM)."""

    def test_huge_title_capped(self, collector_factory):
        from newsbot.collector import MAX_TITLE_CHARS

        rss = (
            b'<?xml version="1.0" encoding="UTF-8"?>'
            b'<rss version="2.0"><channel><title>t</title>'
            b'<item><title>' + b'A' * 100_000 + b'</title>'
            b'<link>http://big.example/1</link></item></channel></rss>'
        )
        collector = collector_factory(
            [{"name": "big", "url": "http://fast.example/feed"}],
            lambda method, url, **kwargs: _FakeStream(_FakeResponse([rss])),
        )
        _, result = next(collector.collect())
        assert len(result.items) == 1
        assert len(result.items[0].title) == MAX_TITLE_CHARS
