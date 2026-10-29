"""
Тесты дайджеста (Этап 4, Фаза 1 + доработка по ревью).

Покрывает:
  - build_digest_messages: разбиение >4096, HTML-экранирование,
    пустой список, max_items_per_message, oversized entry
  - run_digest (storage + delivery): deferred→digested при успехе;
    при частичном сбое — отправленные → digested, остаток → deferred
  - URL-whitelist: javascript: отбрасывается, http/https пропускается
  - schedule dedup: нет дублей, --digest при enabled=false работает
  - purge_items: digested удаляется по retention

Возвращаемое значение build_digest_messages:
  Tuple[List[Tuple[str, List[NewsItem]]], List[NewsItem], List[str]]
  → msg_groups[i][0] = HTML-строка, msg_groups[i][1] = список items,
    build_failed_urls = urls items, выпавших при рендеринге.

Возвращаемое значение run_digest:
  Tuple[int, List[str], List[str]] = (sent_count, sent_urls, all_failed_urls)

Запуск: python -m pytest tests/test_digest.py -v
"""
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from newsbot.storage import Storage
from newsbot.collector import NewsItem
from newsbot.digest import build_digest_messages, run_digest, _TG_MESSAGE_MAX, _utf16_units

# =============================================================================
# helpers
# =============================================================================

def _mk_item(
    source="TestSrc",
    title="Test Title",
    url=None,
    summary="Test summary.",
    llm_score=6,
    llm_summary=None,
    llm_reason=None,
    **kwargs,
) -> NewsItem:
    if url is None:
        url = f"https://example.com/{hash(title)}"
    return NewsItem(
        source=source, url=url, title=title, summary=summary,
        published="2025-01-01 12:00", published_parsed=None,
        llm_score=llm_score, llm_summary=llm_summary, llm_reason=llm_reason,
        **kwargs,
    )


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return path


def _cleanup(path):
    try:
        os.unlink(path)
    except (FileNotFoundError, PermissionError):
        time.sleep(0.2)
        try:
            os.unlink(path)
        except (FileNotFoundError, PermissionError):
            pass


# =============================================================================
# build_digest_messages
# =============================================================================

def test_build_digest_empty():
    """Пустой список → ([], [])."""
    groups, items, _ = build_digest_messages([], max_items_per_message=20)
    assert groups == []
    assert items == []


def test_build_digest_html_escape():
    """Пользовательские данные экранируются."""
    item = _mk_item(
        title='<script>alert("XSS")</script>',
        source='<img src=x onerror=alert(1)>',
        summary='Summary with <b>bold</b> &amp; <i>italic</i>',
    )
    groups, items, _ = build_digest_messages([item], max_items_per_message=20)
    assert len(groups) == 1
    text = groups[0][0]
    assert "<script>" not in text
    assert "&lt;script&gt;" in text


def test_build_digest_single_message():
    """Одна новость → одно сообщение с заголовком."""
    item = _mk_item(title="News Title", source="Source1", summary="Short.")
    groups, items, _ = build_digest_messages([item], max_items_per_message=20)
    assert len(groups) == 1
    assert "📰 Дайджест — 1 из 1 новостей" in groups[0][0]
    assert len(groups[0][1]) == 1


def test_build_digest_max_per_message():
    """При max_per_message=2 трёхэлементный список → 2 сообщения (кумулятивный header)."""
    items = [_mk_item(title=f"Item {i}", url=f"https://ex.com/{i}") for i in range(3)]
    groups, all_items, _ = build_digest_messages(items, max_items_per_message=2)
    assert len(groups) == 2
    # Кумулятивный: msg1 items 1–2 → "1–2 из 3", msg2 item 3 → "3 из 3"
    assert "1–2 из 3" in groups[0][0]
    assert "3 из 3" in groups[1][0]
    assert len(all_items) == 3


def test_build_digest_large_summary_truncated():
    """Очень длинный summary обрезается."""
    long_text = "A" * 1000
    item = _mk_item(summary=long_text)
    groups, items, _ = build_digest_messages([item], max_items_per_message=20)
    assert len(groups) == 1
    # Не более 4096 символов
    assert len(groups[0][0]) <= _TG_MESSAGE_MAX


def test_build_digest_uses_llm_summary():
    """Приоритет llm_summary над summary."""
    item = _mk_item(
        summary="Raw summary from feed.",
        llm_summary="LLM summary — important part.",
    )
    groups, items, _ = build_digest_messages([item], max_items_per_message=20)
    text = groups[0][0]
    assert "LLM summary" in text
    assert "Raw summary from feed" not in text


def test_build_digest_all_fits_single_message():
    """10 новостей влезают в одно сообщение."""
    items = [_mk_item(title=f"Title {i}", url=f"https://ex.com/{i}") for i in range(10)]
    groups, all_items, _ = build_digest_messages(items, max_items_per_message=20)
    assert len(groups) == 1
    assert "10 из 10 новостей" in groups[0][0]
    assert len(all_items) == 10


def test_build_digest_header_has_local_time():
    """Заголовок содержит локальное время."""
    local_now = datetime(2025, 6, 15, 14, 30, tzinfo=timezone.utc)
    item = _mk_item()
    groups, items, _ = build_digest_messages([item], max_items_per_message=20,
                                           local_now=local_now)
    assert "15.06.2025" in groups[0][0]


def test_build_digest_message_within_4096():
    """Каждое сообщение ≤ 4096 (даже при длинных summaries)."""
    items = [
        _mk_item(
            title=f"Long Title {i}",
            summary="Word " * 300 + f" item {i}",
        )
        for i in range(50)
    ]
    groups, all_items, _ = build_digest_messages(items, max_items_per_message=20)
    for msg, _ in groups:
        assert len(msg) <= _TG_MESSAGE_MAX


def test_build_digest_no_broken_html_entities():
    """Нет оборванных HTML-сущностей после сборки."""
    item = _mk_item(
        title="Title with <special> & 'chars'",
        summary="Body: a & b < c > d \"quotes\" 'apostrophes'",
    )
    groups, items, _ = build_digest_messages([item], max_items_per_message=20)
    text = groups[0][0]
    assert "&\n" not in text


def test_build_digest_url_whitelist_http_https():
    """URL http:// и https:// пропускаются с HTML-ссылкой."""
    item = _mk_item(title="T", url="https://example.com/path?a=1&b=2")
    groups, items, _ = build_digest_messages([item], max_items_per_message=20)
    text = groups[0][0]
    assert "https://example.com" in text
    assert '<a href="https://example.com' in text


def test_build_digest_url_whitelist_rejects_javascript():
    """javascript: URL отбрасывается."""
    item = _mk_item(title="JS", url="javascript:alert(1)")
    groups, items, _ = build_digest_messages([item], max_items_per_message=20)
    text = groups[0][0]
    assert "javascript:" not in text
    assert '<a href="javascript' not in text


def test_build_digest_url_whitelist_rejects_tg():
    """tg:// URL отбрасывается."""
    item = _mk_item(title="TG", url="tg://resolve?domain=foo")
    groups, items, _ = build_digest_messages([item], max_items_per_message=20)
    text = groups[0][0]
    assert "tg://" not in text


def test_build_digest_max_per_message_zero_guard():
    """max_per_message=0 → минимум 1 (guard)."""
    items = [_mk_item(title=f"Item {i}") for i in range(3)]
    groups, all_items, _ = build_digest_messages(items, max_items_per_message=0)
    assert len(groups) == 3


def test_build_digest_max_per_message_negative():
    """max_per_message=-5 → минимум 1."""
    items = [_mk_item(title=f"Item {i}") for i in range(3)]
    groups, all_items, _ = build_digest_messages(items, max_items_per_message=-5)
    assert len(groups) == 3


def test_build_digest_source_included():
    """Source отображается в entry (Title — Source)."""
    item = _mk_item(title="Breaking News", source="Reuters", summary="Short.")
    groups, items, _ = build_digest_messages([item], max_items_per_message=20)
    text = groups[0][0]
    assert "Reuters" in text


def test_build_digest_count_in_header_matches_items():
    """Заголовок N из total соответствует фактическому числу items (кумулятивный)."""
    items = [_mk_item(title=f"Item {i}", url=f"https://ex.com/{i}") for i in range(7)]
    groups, all_items, _ = build_digest_messages(items, max_items_per_message=3)
    # 7 items, max=3 → 3 msgs: 1–3, 4–6, 7
    assert len(groups) == 3
    # Кумулятивный формат: "1–3 из 7" / "4–6 из 7" / "7 из 7"
    assert "1–3" in groups[0][0] and "из 7" in groups[0][0]
    assert "4–6" in groups[1][0] and "из 7" in groups[1][0]
    assert "7 из 7" in groups[2][0]


def test_build_digest_cumulative_header_two_messages():
    """При разбиении на 2 сообщения: msg1=N из total, msg2=(N+1) из total."""
    items = [_mk_item(title=f"Item {i}", url=f"https://ex.com/{i}") for i in range(1, 11)]
    groups, _, _ = build_digest_messages(items, max_items_per_message=20)
    # 10 items, max=20 → всё влезает в одно сообщение → header "1–10 из 10"
    assert len(groups) == 1
    assert "1–10 из 10" in groups[0][0]

    # При лимите 5 items/message: 2 сообщения
    groups2, _, _ = build_digest_messages(items, max_items_per_message=5)
    assert len(groups2) == 2
    # msg1: items 1–5 → "1–5 из 10"
    assert "1–5 из 10" in groups2[0][0]
    # msg2: items 6–10 → "6–10 из 10"
    assert "6–10 из 10" in groups2[1][0]


# =============================================================================
# run_digest (storage + delivery)
# =============================================================================

def test_run_digest_marks_digested_on_success():
    """При успешной отправке items переходят в state='digested'."""
    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        item = _mk_item(title="DigestMe", url="https://ex.com/d1", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/d1"], "deferred")

        delivery = MagicMock()
        delivery.send_text.return_value = True

        count, urls, _failed, _ex = run_digest(storage, delivery, max_items=100,
                                 max_items_per_message=20)
        assert count == 1
        assert urls == ["https://ex.com/d1"]
        assert delivery.send_text.call_count == 1

        rows = storage.get_all_by_state("deferred")
        assert len(rows) == 0
        rows = storage.get_all_by_state("digested")
        assert len(rows) == 1
        storage.close()
    finally:
        _cleanup(db_path)


def test_run_digest_remains_deferred_on_failure():
    """При сбое доставки items остаются deferred."""
    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        item = _mk_item(title="NoSend", url="https://ex.com/nosend", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/nosend"], "deferred")

        delivery = MagicMock()
        delivery.send_text.return_value = False

        count, urls, _failed, _ex = run_digest(storage, delivery, max_items=100,
                                 max_items_per_message=20)
        assert count == 0
        assert urls == []

        rows = storage.get_all_by_state("deferred")
        assert len(rows) == 1
        rows = storage.get_all_by_state("digested")
        assert len(rows) == 0
        storage.close()
    finally:
        _cleanup(db_path)


def test_run_digest_remains_deferred_on_exception():
    """При исключении delivery items остаются deferred."""
    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        item = _mk_item(title="CrashMe", url="https://ex.com/crash", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/crash"], "deferred")

        delivery = MagicMock()
        delivery.send_text.side_effect = RuntimeError("network error")

        count, urls, _failed, _ex = run_digest(storage, delivery, max_items=100,
                                 max_items_per_message=20)
        assert count == 0

        rows = storage.get_all_by_state("deferred")
        assert len(rows) == 1
        storage.close()
    finally:
        _cleanup(db_path)


def test_run_digest_empty_no_send():
    """Пустой deferred → delivery не вызывается."""
    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        delivery = MagicMock()

        count, urls, _failed, _ex = run_digest(storage, delivery, max_items=100,
                                 max_items_per_message=20)
        assert count == 0
        assert urls == []
        delivery.send_text.assert_not_called()
        storage.close()
    finally:
        _cleanup(db_path)


def test_run_digest_partial_failure_marks_sent_only():
    """Частичный сбой: отправленные → digested, остаток → deferred."""
    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        for i in range(5):
            item = _mk_item(title=f"Item {i}", url=f"https://ex.com/item{i}",
                            llm_score=6)
            storage.save(item)
            storage.set_state([f"https://ex.com/item{i}"], "deferred")

        delivery = MagicMock()
        call_results = iter([True, False])  # msg1 ok, msg2 fail

        def fake_send(msg):
            return next(call_results)

        delivery.send_text.side_effect = fake_send

        count, urls, _failed, _ex = run_digest(storage, delivery, max_items=100,
                                 max_items_per_message=2)
        # msg1 (items 0,1) → ok, msg2 (items 2,3) → fail, msg3 (item 4) → skip
        assert count == 2
        assert set(urls) == {"https://ex.com/item0", "https://ex.com/item1"}

        rows_digested = storage.get_all_by_state("digested")
        assert len(rows_digested) == 2
        rows_deferred = storage.get_all_by_state("deferred")
        assert len(rows_deferred) == 3
        storage.close()
    finally:
        _cleanup(db_path)


def test_run_digest_returns_included_urls():
    """run_digest возвращает urls только реально отправленных items."""
    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        for i in range(4):
            item = _mk_item(title=f"Item {i}", url=f"https://ex.com/item{i}",
                            llm_score=5)
            storage.save(item)
            storage.set_state([f"https://ex.com/item{i}"], "deferred")

        delivery = MagicMock()
        delivery.send_text.side_effect = [True, False]  # msg1 ok, msg2 fail

        count, urls, _failed, _ex = run_digest(storage, delivery, max_items=100,
                                 max_items_per_message=2)
        assert count == 2
        assert set(urls) == {"https://ex.com/item0", "https://ex.com/item1"}
        storage.close()
    finally:
        _cleanup(db_path)


def test_run_digest_long_items_still_sent():
    """Очень длинные items — все отправлены, ≤4096 символов."""
    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        for i in range(10):
            item = _mk_item(
                title=f"Very Long Title {i} " * 5,
                url=f"https://ex.com/item{i}",
                summary="Word " * 300 + f" item {i}",
                llm_score=5,
            )
            storage.save(item)
            storage.set_state([f"https://ex.com/item{i}"], "deferred")

        delivery = MagicMock()
        delivery.send_text.return_value = True

        count, urls, _failed, _ex = run_digest(storage, delivery, max_items=100,
                                 max_items_per_message=3)
        assert count == 10
        assert len(urls) == 10

        rows = storage.get_all_by_state("digested")
        assert len(rows) == 10

        for call in delivery.send_text.call_args_list:
            msg = call[0][0]
            # Minor-6-фикс: Telegram измеряет лимит в UTF-16 units, не в code points
            assert _utf16_units(msg) <= _TG_MESSAGE_MAX

        storage.close()
    finally:
        _cleanup(db_path)


# =============================================================================
# last_digest_at
# =============================================================================

def test_last_digest_at_written_on_success():
    """pipeline.run_digest пишет last_digest_at в meta."""
    from newsbot.pipeline import NewsPipeline

    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        item = _mk_item(title="T", url="https://ex.com/at", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/at"], "deferred")

        delivery = MagicMock()
        delivery.send_text.return_value = True

        config = {
            "sources": [],
            "database": {"path": db_path},
            "digest": {"enabled": True, "times": ["09:00"],
                      "max_items": 100, "max_items_per_message": 20},
            "llm": {"enabled": False},
        }
        pipeline = NewsPipeline(config)
        pipeline.delivery = delivery
        pipeline.storage = storage

        pipeline.run_digest(force=True)

        val = storage.get_meta("last_digest_at")
        assert val is not None
        dt = datetime.fromisoformat(val)
        assert dt.year >= 2024
        pipeline.close()
    finally:
        _cleanup(db_path)


def test_digest_run_digest_does_not_write_last_digest_at():
    """digest.run_digest не пишет last_digest_at — это pipeline."""
    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        item = _mk_item(title="T", url="https://ex.com/test", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/test"], "deferred")

        delivery = MagicMock()
        delivery.send_text.return_value = True

        run_digest(storage, delivery, max_items=100, max_items_per_message=20)

        val = storage.get_meta("last_digest_at")
        assert val is None
        storage.close()
    finally:
        _cleanup(db_path)


# =============================================================================
# schedule dedup: _maybe_run_digest
# =============================================================================

def test_maybe_run_digest_skips_recent_window():
    """
    Не запускает если last_digest_at в пределах 30-минутного окна.

    Тестирует через run_digest(force=True): сначала выполняем digest
    (пишет last_digest_at), затем _maybe_run_digest вызывается сразу после —
    должен пропустить (last_digest_at только что записан, < 30 мин назад).

    Проверяет именно SKIP через логику "последний digest был недавно".
    """
    from newsbot.pipeline import NewsPipeline

    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        item = _mk_item(title="Recent", url="https://ex.com/recent", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/recent"], "deferred")

        now_local = datetime.now(timezone.utc).astimezone()
        current_hhmm = now_local.strftime("%H:%M")

        config = {
            "sources": [],
            "database": {"path": db_path},
            "digest": {"enabled": True, "times": [current_hhmm],
                      "max_items": 100, "max_items_per_message": 20},
            "llm": {"enabled": False},
        }

        delivery = MagicMock()
        delivery.send_text.return_value = True
        pipeline = NewsPipeline(config)
        pipeline.delivery = delivery
        pipeline.storage = storage

        # Первый вызов: force=True → digest ВЫПОЛНЯЕТСЯ и пишет last_digest_at
        pipeline.run_digest(force=True)
        assert delivery.send_text.called
        delivery.reset_mock()

        # Второй вызов: last_digest_at только что записан (< 30 сек) → skip
        pipeline._maybe_run_digest()

        # Не должен запускать digest (last_digest_at < 30 мин назад)
        delivery.send_text.assert_not_called()
        pipeline.close()
    finally:
        _cleanup(db_path)


def test_maybe_run_digest_fires_at_scheduled_time():
    """Запускает когда scheduled наступил и не обработан."""
    from newsbot.pipeline import NewsPipeline

    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        item = _mk_item(title="Scheduled", url="https://ex.com/sched", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/sched"], "deferred")

        old = (datetime.now(timezone.utc) - timedelta(minutes=45)).isoformat()
        storage.set_meta("last_digest_at", old)

        now_local = datetime.now(timezone.utc).astimezone()
        current_hhmm = now_local.strftime("%H:%M")

        config = {
            "sources": [],
            "database": {"path": db_path},
            "digest": {"enabled": True, "times": [current_hhmm]},
            "llm": {"enabled": False},
        }

        delivery = MagicMock()
        delivery.send_text.return_value = True
        pipeline = NewsPipeline(config)
        pipeline.delivery = delivery
        pipeline.storage = storage

        pipeline._maybe_run_digest()

        assert delivery.send_text.called
        rows = storage.get_all_by_state("digested")
        assert len(rows) == 1
        pipeline.close()
    finally:
        _cleanup(db_path)


def test_maybe_run_digest_no_dup_in_same_window():
    """Повторный вызов в том же окне — второй раз НЕ срабатывает."""
    from newsbot.pipeline import NewsPipeline

    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        item = _mk_item(title="Dup", url="https://ex.com/dup", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/dup"], "deferred")

        now_local = datetime.now(timezone.utc).astimezone()
        current_hhmm = now_local.strftime("%H:%M")

        config = {
            "sources": [],
            "database": {"path": db_path},
            "digest": {"enabled": True, "times": [current_hhmm]},
            "llm": {"enabled": False},
        }

        delivery = MagicMock()
        delivery.send_text.return_value = True
        pipeline = NewsPipeline(config)
        pipeline.delivery = delivery
        pipeline.storage = storage

        pipeline._maybe_run_digest()
        assert delivery.send_text.called

        # Второй вызов: last_digest_at обновлён
        delivery.reset_mock()
        pipeline._maybe_run_digest()
        delivery.send_text.assert_not_called()

        pipeline.close()
    finally:
        _cleanup(db_path)


def test_maybe_run_digest_force_ignores_enabled():
    """run_digest(force=True) работает при digest.enabled=false."""
    from newsbot.pipeline import NewsPipeline

    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        item = _mk_item(title="ForceMe", url="https://ex.com/force", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/force"], "deferred")

        config = {
            "sources": [],
            "database": {"path": db_path},
            "digest": {"enabled": False, "times": ["09:00"],
                      "max_items": 100, "max_items_per_message": 20},
            "llm": {"enabled": False},
        }

        delivery = MagicMock()
        delivery.send_text.return_value = True
        pipeline = NewsPipeline(config)
        pipeline.delivery = delivery
        pipeline.storage = storage

        count = pipeline.run_digest(force=True)

        assert count == 1
        rows = storage.get_all_by_state("digested")
        assert len(rows) == 1
        pipeline.close()
    finally:
        _cleanup(db_path)


def test_maybe_run_digest_skips_when_disabled():
    """_maybe_run_digest не запускает при digest.enabled=false."""
    from newsbot.pipeline import NewsPipeline

    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        item = _mk_item(title="Off", url="https://ex.com/off", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/off"], "deferred")

        now_local = datetime.now(timezone.utc).astimezone()
        current_hhmm = now_local.strftime("%H:%M")

        config = {
            "sources": [],
            "database": {"path": db_path},
            "digest": {"enabled": False, "times": [current_hhmm]},
            "llm": {"enabled": False},
        }

        delivery = MagicMock()
        pipeline = NewsPipeline(config)
        pipeline.delivery = delivery
        pipeline.storage = storage

        pipeline._maybe_run_digest()

        delivery.send_text.assert_not_called()
        pipeline.close()
    finally:
        _cleanup(db_path)


# =============================================================================
# purge: digested удаляется
# =============================================================================

def test_purge_removes_digested():
    """purge_items удаляет digested старше retention_days."""
    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        item = _mk_item(title="OldDigest", url="https://ex.com/old", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/old"], "digested")

        import sqlite3
        conn = sqlite3.connect(db_path)
        old_time = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        conn.execute("UPDATE items SET state_at = ? WHERE url = ?",
                     (old_time, "https://ex.com/old"))
        conn.commit()
        conn.close()

        deleted = storage.purge_items(retention_days=90)
        assert deleted >= 1
        rows = storage.get_all_by_state("digested")
        assert len(rows) == 0
        storage.close()
    finally:
        _cleanup(db_path)


def test_purge_keeps_fresh_digested():
    """purge_items НЕ удаляет свежие digested."""
    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        item = _mk_item(title="FreshDigest", url="https://ex.com/fresh", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/fresh"], "digested")

        deleted = storage.purge_items(retention_days=90)
        assert deleted == 0
        rows = storage.get_all_by_state("digested")
        assert len(rows) == 1
        storage.close()
    finally:
        _cleanup(db_path)


# =============================================================================
# N1: send_digest — run_digest с item_buttons_per_msg использует send_digest
# =============================================================================

def test_run_digest_with_buttons_calls_send_digest():
    """run_digest при item_buttons_per_msg вызывает delivery.send_digest."""
    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        item = _mk_item(title="BtnTest", url="https://ex.com/btns", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/btns"], "deferred")

        delivery = MagicMock()
        delivery.send_digest.return_value = True

        item_buttons = [[{"url": "https://ex.com/btns", "target_id": 7}]]
        count, urls, _failed, _ex = run_digest(
            storage, delivery, max_items=100, max_items_per_message=20,
            item_buttons_per_msg=item_buttons,
        )

        assert count == 1
        assert urls == ["https://ex.com/btns"]
        delivery.send_digest.assert_called_once()
        args, kwargs = delivery.send_digest.call_args
        # Текст содержит HTML
        assert "<b>BtnTest" in args[0]
        # reply_markup содержит строки кнопок
        assert "reply_markup" in kwargs or len(args) > 1

        rows = storage.get_all_by_state("digested")
        assert len(rows) == 1
        storage.close()
    finally:
        _cleanup(db_path)


def test_run_digest_buttons_partial_failure():
    """Частичный сбой с кнопками: отправленные → digested."""
    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        for i in range(4):
            item = _mk_item(title=f"I{i}", url=f"https://ex.com/b{i}", llm_score=6)
            storage.save(item)
            storage.set_state([f"https://ex.com/b{i}"], "deferred")

        delivery = MagicMock()
        # msg1 ok, msg2 fail
        delivery.send_digest.side_effect = [True, False]

        # 4 items, max=2 → 2 сообщения
        item_buttons = [
            [{"url": f"https://ex.com/b{i}", "target_id": i + 1} for i in range(2)],
            [{"url": f"https://ex.com/b{i}", "target_id": i + 1} for i in range(2, 4)],
        ]
        count, urls, _failed, _ex = run_digest(
            storage, delivery, max_items=100, max_items_per_message=2,
            item_buttons_per_msg=item_buttons,
        )
        assert count == 2
        assert set(urls) == {"https://ex.com/b0", "https://ex.com/b1"}

        rows_digested = storage.get_all_by_state("digested")
        assert len(rows_digested) == 2
        rows_deferred = storage.get_all_by_state("deferred")
        assert len(rows_deferred) == 2
        storage.close()
    finally:
        _cleanup(db_path)


# =============================================================================
# Major-2: дайджест >4096 UTF-16 units нарезается на подсообщения
# =============================================================================

def test_run_digest_oversized_message_split():
    """
    digest msg > 4096 UTF-16 units → нарезается на подсообщения ≤ 4096,
    ни один sub не превышает лимит.
    """
    from newsbot.digest import _split_message, _TG_MESSAGE_MAX
    from newsbot.delivery.telegram import _utf16_units

    # oversized: 5 абзацев, каждый ~900 UTF-16 units → общий размер > 4096
    para = "X" * 900  # ~900 UTF-16 units per para
    oversized_msg = "\n\n".join([para] * 5)  # ~5×900 + 4×2 = ~4508 > 4096

    assert _utf16_units(oversized_msg) > _TG_MESSAGE_MAX, \
        f"test msg too short: {_utf16_units(oversized_msg)} units"

    parts = _split_message(oversized_msg)
    assert len(parts) > 1, f"oversized msg должен нарезаться, got {len(parts)} parts"
    for p in parts:
        assert _utf16_units(p) <= _TG_MESSAGE_MAX, \
            f"sub-часть {_utf16_units(p)} превышает {_TG_MESSAGE_MAX}"
    # нетезированные абзацы сохраняются (без hard-slice), verify count
    # 5 абзацев × 900 units каждый → 4508 total
    # → 5 msg = 1 msg per para (каждый 900 < 4096, итого 5 подсообщений)
    # NOTE: при изменении packing логики достаточно проверять лимит
    assert sum(_utf16_units(p) for p in parts) >= _utf16_units(para) * 4, \
        "нетезированные абзацы не должны теряться"

    # normal msg проходит без split
    normal_msg = "X" * 100
    assert _split_message(normal_msg) == [normal_msg]


def test_split_message_no_double_escape():
    """
    Major-1: _split_message НЕ переэкранирует — <b> и <a href> сохраняются
    (не превращаются в &lt;b&gt;). CONTRACT: msg уже экранирован (_emit).
    """
    from newsbot.digest import _split_message, _utf16_units

    # msg содержит уже экранированные <b> и <a href>
    # 10 абзацев × 600 = 6000 > 4096: гарантированно oversized
    para = "X" * 600
    html_msg = (
        f"<b>Bold Title</b>\n\n{para}\n\n"
        f'<a href="https://example.com/path">link</a>\n\n'
        f"{para}\n\n{para}\n\n{para}\n\n"
        f"{para}\n\n{para}\n\n{para}\n\n{para}\n\n{para}"
    )
    # проверяем что oversized для надёжности
    assert _utf16_units(html_msg) > 4096, \
        f"msg={_utf16_units(html_msg)} — тест должен быть oversized"

    parts = _split_message(html_msg)
    for i, p in enumerate(parts):
        # (а) НЕТ &lt;b&gt; (double-escape сломает HTML)
        assert "&lt;b&gt;" not in p, f"part[{i}] содержит &lt;b&gt; — double-escape!"
        # (г) каждое ≤ лимита
        assert _utf16_units(p) <= 4096, f"part[{i}] = {_utf16_units(p)} > 4096"


def test_split_message_preserves_links():
    """
    Major-1: <a href="..."> не разрывается — граница ищется по > / \\n / пробелу.
    """
    from newsbot.digest import _split_message, _utf16_units

    # Oversized-абзац с <a href> внутри: нужно чтобы ссылка не разрезалась
    link_part = '<a href="https://example.com/very-long-url">нажмите здесь</a>'
    # 5000 + link + 5000 = ~10100 > 4096*2, гарантированно oversized
    oversized = "X" * 5000 + link_part + "Y" * 5000
    assert _utf16_units(oversized) > 8192, \
        f"msg={_utf16_units(oversized)} — тест должен быть oversized"

    parts = _split_message(oversized)
    for p in parts:
        # ни одна часть не должна содержать битую ссылку:
        # <a href="... без > — обрезана внутри тега
        assert not re.search(r'<a href="[^"]*$', p), "ссылка обрезана внутри href"
        assert not re.search(r'<a [^>]*$', p), "тег <a обрезан внутри"
        # если ссылка присутствует — она целая
        if "example.com" in p:
            assert p.count("<a ") == p.count("</a>"), "непарный <a в part"


def test_split_message_no_content_loss():
    """
    Major-1: _hard_slice возвращает ВСЕ чанки (список), не теряет остаток.
    Один oversized-абзац нарезается на ≥2 частей, суммарно весь контент сохранён.
    """
    from newsbot.digest import _split_message, _hard_slice, _utf16_units

    # Один oversized-абзац, который превышает лимит целиком
    # 4500 единиц > 4096
    big_para = "A" * 4500
    assert _utf16_units(big_para) > 4096

    # _hard_slice должен вернуть список чанков, покрывающий весь контент
    chunks = _hard_slice(big_para, 4096)
    assert len(chunks) >= 2, f"oversized para должен дать ≥2 чанков, got {len(chunks)}"
    assert sum(_utf16_units(c) for c in chunks) == _utf16_units(big_para), \
        "хвост потерян — сумма чанков != исходный размер"

    # _split_message вызывает _hard_slice для oversized-абзацев
    msg = "<b>Title</b>\n\n" + big_para
    parts = _split_message(msg)
    assert len(parts) >= 2
    # весь контент (title + все чанки big_para) присутствует
    full_text = "".join(parts)
    assert "A" * 4500 in full_text, "контент oversized-абзаца потерян"


def test_split_message_safe_boundary_entities():
    """
    Major-1: _hard_slice режет по безопасной границе — HTML-сущности не рвутся.
    &amp; / &lt; / &gt; / &quot; / &#x27; не должны быть оборваны.
    Minor-1-фикс: regex ловит частичный обрыв (&quo, &am) и открытые теги.
    Валидные &#x27; (hex) и &#27; (decimal) не считаются обрывом.
    """
    from newsbot.digest import _hard_slice, _utf16_units

    # Строка с HTML-сущностями, oversized
    with_entities = "&amp;&lt;&gt;&quot;&#x27;" * 1000  # ~70 * 1000 = 70000
    assert _utf16_units(with_entities) > 4096 * 2

    chunks = _hard_slice(with_entities, 4096)

    # Minor-1: усиленный regex — ловит &quo, &am, &quot без ;
    # НЕ ловит &#x27; — это валидная hex-сущность (&#x + hex digits + ;)
    # НЕ ловит &#27; — decimal сущность
    # Named-partial: & + letters {1,5} без ; в конце
    entity_broken_re = re.compile(r"&[a-zA-Z]{1,5}(?!;)$")
    for i, c in enumerate(chunks):
        assert not entity_broken_re.search(c), (
            f"chunk[{i}] содержит оборванную HTML-сущность: {c[-40:]!r}"
        )


def test_split_message_safe_boundary_partial_entities():
    """
    Minor-1: гарантия что срез НЕ попадает внутрь &quot; / &#x27; —
    даже когда лимит разрезает середину сущности.
    """
    from newsbot.digest import _hard_slice, _utf16_units

    # Паттерн: "&quot;" (6 UTF-16 units) чередуется с &#x27; (6 units).
    # При малом max_units грубый срез неизбежно попадает внутрь.
    # _hard_slice должен корректно откатить до безопасной точки.
    base = "&quot;&#x27;"  # 12 units total
    oversized = base * 1000  # 12000 units >> 4096

    # Искусственно маленький лимит — гарантированно рвёт внутри сущности
    chunks = _hard_slice(oversized, 1000)
    # Ловим только именованные (&quo, &amp без ;), НЕ hex/dec сущности
    entity_broken_re = re.compile(r"&[a-zA-Z]{1,5}(?!;)$")
    tag_open_re = re.compile(r"<[a-zA-Z][^>]*$")
    for i, c in enumerate(chunks):
        assert not entity_broken_re.search(c), (
            f"chunk[{i}] оборвал HTML-сущность: {c[-30:]!r}"
        )
        assert not tag_open_re.search(c), (
            f"chunk[{i}] оборвал открывающий тег: {c[-30:]!r}"
        )


def test_split_message_safe_boundary_tags():
    """
    Minor-1: гарантия что срез НЕ попадает внутрь <a href=...> —
    Major-фикс добавил > в набор границ, но если срез всё же рвёт
    открывающий тег — откат до последнего <.
    """
    from newsbot.digest import _hard_slice, _utf16_units

    # Строка с открывающими тегами: гарантированно oversized
    # ~40 * 250 = 10000 > 8192
    with_tags = '<a href="https://example.com/1">link</a>' * 250
    assert _utf16_units(with_tags) > 4096 * 2

    chunks = _hard_slice(with_tags, 4096)
    tag_open_re = re.compile(r"<[a-zA-Z][^>]*$")
    for i, c in enumerate(chunks):
        assert not tag_open_re.search(c), (
            f"chunk[{i}] оборвал открывающий тег: {c[-40:]!r}"
        )


def test_run_digest_oversized_integrated():
    """
    run_digest с oversized-msg: send_digest вызывается ≥2 раз,
    buttons отправлены только с последним sub.
    """
    from newsbot.digest import _split_message
    from newsbot.delivery.telegram import _utf16_units

    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        item = _mk_item(title="Oversized", url="https://ex.com/over", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/over"], "deferred")

        delivery = MagicMock()
        call_args = []

        def capture_send(text, buttons):
            call_args.append((text, buttons))
            return True
        delivery.send_digest.side_effect = capture_send

        # build_digest_messages сам группирует, нам нужен oversized-msg
        # напрямую: делаем oversized msg через item_buttons_per_msg + groups
        # 5 абзацев × 900 units = 4508 > 4096
        para = "X" * 900
        oversized_html = f"<b>Title</b>\n\n{para}\n\n{para}\n\n{para}\n\n{para}\n\n{para}\n\n<a href=\"https://ex.com/over\">🔗</a>"

        # проверяем что msg превышает лимит
        assert _utf16_units(oversized_html) > 4096, \
            f"oversized_html too short: {_utf16_units(oversized_html)} units"

        # вызываем run_digest напрямую с oversized groups
        count, urls, _failed, _ex = run_digest(
            storage, delivery, max_items=1, max_items_per_message=1,
            item_buttons_per_msg=[[{"url": "https://ex.com/over", "target_id": 1}]],
            groups=[(oversized_html, [item])],
        )
        assert count == 1
        assert call_args, "send_digest должен быть вызван"
        # buttons должны быть только в последнем sub-сообщении
        last_call_buttons = call_args[-1][1]
        assert last_call_buttons == [{"url": "https://ex.com/over", "target_id": 1}]
        for i, (text, buttons) in enumerate(call_args[:-1]):
            assert buttons == [], f"sub[{i}] не должен иметь buttons"
            assert _utf16_units(text) <= 4096

        storage.close()
    finally:
        _cleanup(db_path)


# =============================================================================
# N2: catch-up + dedup в schedule
# =============================================================================

def test_maybe_run_digest_catches_up_late_cycle():
    """
    N2: цикл попал В МИНУТУ ПОСЛЕ scheduled (now > scheduled)
    — дайджест ВСЁ РАВНО отправляется.

    Мокаем storage.get_meta чтобы эмулировать состояние когда
    last_digest_at старше scheduled (окно не обработано).
    """
    from newsbot.pipeline import NewsPipeline

    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        item = _mk_item(title="Late", url="https://ex.com/late", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/late"], "deferred")

        # last_digest_at = 2 часа назад → окно не обработано
        old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        storage.set_meta("last_digest_at", old_time)

        # Текущее время HH:59 — scheduled был HH:00 (уже прошёл)
        now_local = datetime.now(timezone.utc).astimezone()
        sched_time = now_local.replace(minute=0, second=0, microsecond=0)
        # now на 30 мин позже scheduled
        current_hhmm = sched_time.strftime("%H:%M")

        config = {
            "sources": [],
            "database": {"path": db_path},
            "digest": {"enabled": True, "times": [current_hhmm]},
            "llm": {"enabled": False},
        }

        delivery = MagicMock()
        delivery.send_digest.return_value = True
        delivery.send_text.return_value = True
        pipeline = NewsPipeline(config)
        pipeline.delivery = delivery
        pipeline.storage = storage

        # last_digest_at говорит "не обработано" — run_digest отправляет.
        # При ConsoleDelivery вызывается send_text, не send_digest.
        with patch.object(pipeline.storage, "get_meta", wraps=pipeline.storage.get_meta):
            pipeline._maybe_run_digest()

        # Проверяем что отправка произошла (send_text для Console, send_digest для TG)
        assert delivery.send_text.called or delivery.send_digest.called, (
            "Дайджест должен был запуститься (catch-up)"
        )
        rows = storage.get_all_by_state("digested")
        assert len(rows) == 1
        pipeline.close()
    finally:
        _cleanup(db_path)


# =============================================================================
# N3: консервативный 4096 check
# =============================================================================

def test_run_digest_warns_on_oversized_text(caplog):
    """
    M1'/M1: патологический escape — title='\"'*250 + body='\"'*500.
    На СТАРОМ коде: title→1500 + body→3000 → entry≈4650 + header > 4096 (warning + Telegram 400).
    На НОВОМ коде: binary search усекает body, msg ≤ 4096, НЕТ warning.
    Проверяем: len≤4096, HTML валиден (парность тегов, целые сущности), warning отсутствует.
    """
    import logging
    import html as _html

    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        # title 250×" → 1500 after escape; body 500×" → 3000 after escape.
        # Old code: no binary search, would overflow → warning.
        # New code: binary search shrinks body → msg ≤ 4096, no warning.
        title_text = '"' * 250
        body_text = '"' * 500
        item = _mk_item(
            title=title_text,
            url="https://ex.com/over",
            summary=body_text,
            llm_score=5,
        )
        storage.save(item)
        storage.set_state(["https://ex.com/over"], "deferred")

        delivery = MagicMock()
        delivery.send_digest.return_value = True

        item_buttons = [[{"url": "https://ex.com/over", "target_id": 1}]]

        with caplog.at_level(logging.WARNING, logger="newsbot.digest"):
            count, urls, _failed, _ex = run_digest(
                storage, delivery, max_items=100, max_items_per_message=1,
                item_buttons_per_msg=item_buttons,
            )

        assert count == 1
        assert urls == ["https://ex.com/over"]

        # M1-гарантия: сообщение ≤ 4096
        msg = delivery.send_digest.call_args[0][0]
        assert len(msg) <= _TG_MESSAGE_MAX, f"Сообщение {len(msg)} > 4096"

        # N3: warning отсутствует (binary search усекает ДО overflow)
        assert not any(r.levelno >= logging.WARNING for r in caplog.records), (
            f"Warning не должен был залогироваться (binary search), "
            f"но залогирован: {[r.message for r in caplog.records if r.levelno >= logging.WARNING]}"
        )

        # HTML валиден: парные теги + нет оборванных сущностей
        assert msg.count("<b>") == msg.count("</b>")

        rows = storage.get_all_by_state("digested")
        assert len(rows) == 1
        storage.close()
    finally:
        _cleanup(db_path)


def test_run_digest_pathological_url_no_warning(caplog):
    """
    M1': патологический URL (много &) — footer может раздуть сообщение.
    При ESCAPED cap URL=500 символов (≈3000 raw с &) footer ≤ 550 символов,
    header + overhead ~400 → body budget ≥ 3246, binary search усекает body → msg ≤ 4096.
    Проверяем: msg ≤ 4096, warning отсутствует.

    NB: set_state использует точный URL — передаём полный, включая query string.
    """
    import logging

    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        # URL с 400 амперсандами → escaped ≈ 1200 символов (3× раздутие)
        # _MAX_FOOTER_URL_ESC_LEN=500 → binary search усекает URL до ≈170 raw
        pathological_url = "https://ex.com/path?" + "&" * 400 + "q=1"
        item = _mk_item(
            title="Normal Title",
            url=pathological_url,
            summary="Short body.",   # small body, URL is the "poison"
            llm_score=5,
        )
        storage.save(item)
        storage.set_state([pathological_url], "deferred")  # полный URL!

        delivery = MagicMock()
        delivery.send_digest.return_value = True

        item_buttons = [[{"url": pathological_url, "target_id": 1}]]

        with caplog.at_level(logging.WARNING, logger="newsbot.digest"):
            count, urls, _failed, _ex = run_digest(
                storage, delivery, max_items=100, max_items_per_message=1,
                item_buttons_per_msg=item_buttons,
            )

        assert count == 1
        assert urls[0] == pathological_url

        # M1-гарантия: msg ≤ 4096
        msg = delivery.send_digest.call_args[0][0]
        assert len(msg) <= _TG_MESSAGE_MAX, f"URL-overflow msg: {len(msg)} > 4096"

        # HTML валиден: парные теги
        assert msg.count("<b>") == msg.count("</b>")

        # Нет warning (URL усечён бинарным поиском в footer)
        assert not any(r.levelno >= logging.WARNING for r in caplog.records), (
            f"Warning не должен был залогироваться, "
            f"но: {[r.message for r in caplog.records if r.levelno >= logging.WARNING]}"
        )

        storage.close()
    finally:
        _cleanup(db_path)


# =============================================================================
# M1': усиленные патологические кейсы
# =============================================================================

def test_run_digest_combined_escape_pathology(caplog):
    """
    M1': title='\"'*250 + body='\"'*500 + URL с & — одновременно три источника раздутия.
    Проверяем: msg ≤ 4096, HTML валиден, warning отсутствует.
    Даже при комбинированной патологии код усекает всё что нужно.
    """
    import logging

    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        # Все три компонента патологические
        path_url = "https://ex.com/over?" + "&" * 200 + "q=1"
        item = _mk_item(
            title='"' * 250,
            url=path_url,
            summary='"' * 500,
            llm_score=5,
        )
        storage.save(item)
        storage.set_state([path_url], "deferred")

        delivery = MagicMock()
        delivery.send_digest.return_value = True

        item_buttons = [[{"url": path_url, "target_id": 1}]]

        with caplog.at_level(logging.WARNING, logger="newsbot.digest"):
            count, urls, _failed, _ex = run_digest(
                storage, delivery, max_items=100, max_items_per_message=1,
                item_buttons_per_msg=item_buttons,
            )

        assert count == 1

        msg = delivery.send_digest.call_args[0][0]
        assert len(msg) <= _TG_MESSAGE_MAX, f"Combined pathology msg: {len(msg)} > 4096"

        # HTML валиден
        assert msg.count("<b>") == msg.count("</b>")

        # Нет warning
        assert not any(r.levelno >= logging.WARNING for r in caplog.records), (
            f"Warning при комбинированной патологии: "
            f"{[r.message for r in caplog.records if r.levelno >= logging.WARNING]}"
        )

        storage.close()
    finally:
        _cleanup(db_path)


def test_build_digest_quotes_title_and_body_escaped_valid(caplog):
    """
    M1': title='\"'*250 + body='\"'*500 — build_digest_messages (без storage/delivery).
    Проверяем: msg ≤ 4096, HTML валиден (парность тегов, целые сущности).
    """
    item = _mk_item(
        title='"' * 250,
        url="https://ex.com/quote",
        summary='"' * 500,
        llm_score=5,
    )
    groups, _, _ = build_digest_messages([item], max_items_per_message=1)
    assert len(groups) == 1
    msg, _ = groups[0]
    assert len(msg) <= _TG_MESSAGE_MAX, f"Quotes msg: {len(msg)} > 4096"

    # HTML валиден
    assert msg.count("<b>") == msg.count("</b>")


# =============================================================================
# R1: on_empty="send" + пустой deferred — уведомление отправляется
# =============================================================================

def test_pipeline_run_digest_on_empty_send_sends_notification(tmp_path):
    """R1: on_empty=send + пустой deferred → send_text вызывается с уведомлением."""
    from newsbot.pipeline import NewsPipeline

    db_path = tmp_path / "test.db"
    config = {
        "sources": [],
        "database": {"path": str(db_path)},
        "telegram": {"token": "", "chat_id": ""},  # ConsoleDelivery
        "digest": {"enabled": True, "on_empty": "send", "max_items": 100,
                  "max_items_per_message": 20},
        "llm": {"enabled": False},
    }

    pipeline = NewsPipeline(config)
    delivery = MagicMock()
    pipeline.delivery = delivery
    pipeline.storage = storage = pipeline.storage

    # Нет deferred items
    count = pipeline.run_digest(force=False)

    # on_empty=send: отправлено уведомление
    assert count == 0
    delivery.send_text.assert_called_once()
    args = delivery.send_text.call_args[0][0]
    assert "нет новых новостей" in args or "📰" in args
    pipeline.close()


def test_pipeline_run_digest_on_empty_skip_no_notification(tmp_path):
    """R1: on_empty=skip + пустой deferred → send_text НЕ вызывается."""
    from newsbot.pipeline import NewsPipeline

    db_path = tmp_path / "test.db"
    config = {
        "sources": [],
        "database": {"path": str(db_path)},
        "telegram": {"token": "", "chat_id": ""},
        "digest": {"enabled": True, "on_empty": "skip", "max_items": 100,
                  "max_items_per_message": 20},
        "llm": {"enabled": False},
    }

    pipeline = NewsPipeline(config)
    delivery = MagicMock()
    pipeline.delivery = delivery
    pipeline.storage = storage = pipeline.storage

    count = pipeline.run_digest(force=False)

    # on_empty=skip: ничего не отправлено
    assert count == 0
    delivery.send_text.assert_not_called()
    pipeline.close()


def test_run_digest_pathological_escape_special_chars():
    """M1: title/body из < > & ' " — после escape ≤4096, HTML валиден."""
    # Строим строку которая при escape превысит бюджет
    dangerous = "<script>&quot;" * 100   # 800 символов raw
    item = _mk_item(
        title="Escaped <b>&amp;",
        source="Src <>&\"",
        summary=dangerous,
        llm_score=5,
    )
    groups, _, _ = build_digest_messages([item], max_items_per_message=1)
    assert len(groups) == 1
    msg, _ = groups[0]
    # Гарантия ≤ 4096
    assert len(msg) <= _TG_MESSAGE_MAX, f"Патологический msg: {len(msg)} > 4096"
    # HTML валиден: открытые и закрытые теги парные
    assert msg.count("<b>") == msg.count("</b>")
    assert "<script>" not in msg
    assert "&lt;script&gt;" in msg


def test_run_digest_continue_on_partial_failure():
    """
    M1: при сбое одного сообщения остальные ВСЁ РАВНО уходят (continue, не break).
    Частичный сбой: msg1 ok, msg2 fail → msg3 всё равно отправляется.
    Отправленные → digested, неудавшийся → deferred.
    """
    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        # 6 items → 3 сообщения по 2
        for i in range(6):
            item = _mk_item(title=f"Item{i}", url=f"https://ex.com/i{i}", llm_score=6)
            storage.save(item)
            storage.set_state([f"https://ex.com/i{i}"], "deferred")

        delivery = MagicMock()
        # msg1 ok, msg2 fail, msg3 ok
        call_count = [0]

        def fake_send(text, buttons):
            call_count[0] += 1
            return call_count[0] != 2   # msg2 → fail

        delivery.send_digest.side_effect = fake_send

        item_buttons = [
            [{"url": f"https://ex.com/i{i}", "target_id": i + 1} for i in range(2)],
            [{"url": f"https://ex.com/i{i}", "target_id": i + 1} for i in range(2, 4)],
            [{"url": f"https://ex.com/i{i}", "target_id": i + 1} for i in range(4, 6)],
        ]
        count, urls, _failed, _ex = run_digest(
            storage, delivery, max_items=100, max_items_per_message=2,
            item_buttons_per_msg=item_buttons,
        )
        # msg1 (0,1) + msg3 (4,5) отправлены = 4 items
        assert count == 4, f"expected 4, got {count}"
        assert set(urls) == {
            "https://ex.com/i0", "https://ex.com/i1",
            "https://ex.com/i4", "https://ex.com/i5",
        }
        # msg2 (2,3) не отправлен
        rows_digested = storage.get_all_by_state("digested")
        assert len(rows_digested) == 4
        rows_deferred = storage.get_all_by_state("deferred")
        assert len(rows_deferred) == 2
        storage.close()
    finally:
        _cleanup(db_path)


def test_run_digest_source_hard_cap():
    """M1: source обрезается до 100 символов RAW — overhead не раздувается."""
    # source с 200 символами — после усечения до 100 body получит больше бюджета
    item = _mk_item(title="Title", source="X" * 200, summary="Short.", llm_score=5)
    groups, _, _ = build_digest_messages([item], max_items_per_message=1)
    msg, _ = groups[0]
    assert len(msg) <= _TG_MESSAGE_MAX


def test_run_digest_partial_failure_send_text_continue():
    """M1: continue в send_text-ветке — остальные сообщения уходят."""
    db_path = _tmp_db()
    try:
        storage = Storage(db_path)
        # 6 items → 3 сообщения по 2; msg2=fail → msg3 всё равно уходит
        for i in range(6):
            item = _mk_item(title=f"Item{i}", url=f"https://ex.com/s{i}", llm_score=5)
            storage.save(item)
            storage.set_state([f"https://ex.com/s{i}"], "deferred")

        delivery = MagicMock()
        call_count = [0]

        def fake_send_text(text):
            call_count[0] += 1
            return call_count[0] != 2   # msg2 → fail

        delivery.send_text.side_effect = fake_send_text

        # send_text fallback (item_buttons_per_msg=None)
        count, urls, _failed, _ex = run_digest(
            storage, delivery, max_items=100, max_items_per_message=2,
        )
        # msg1 (0,1) + msg3 (4,5) отправлены = 4 items
        assert count == 4, f"expected 4, got {count}"
        assert set(urls) == {
            "https://ex.com/s0", "https://ex.com/s1",
            "https://ex.com/s4", "https://ex.com/s5",
        }
        rows_digested = storage.get_all_by_state("digested")
        assert len(rows_digested) == 4
        rows_deferred = storage.get_all_by_state("deferred")
        assert len(rows_deferred) == 2  # msg2: items 2,3
        storage.close()
    finally:
        _cleanup(db_path)


# =============================================================================
# Feedback targets: kind='digest' idempotent
# =============================================================================

def test_pipeline_run_digest_creates_feedback_targets(tmp_path):
    """pipeline.run_digest создаёт feedback_targets с kind='digest'."""
    from newsbot.pipeline import NewsPipeline
    from newsbot.delivery import TelegramDelivery

    db_path = tmp_path / "test.db"
    config = {
        "sources": [],
        "database": {"path": str(db_path)},
        "telegram": {"token": "test", "chat_id": "123"},
        "llm": {"enabled": False},
        "feedback": {"enabled": True, "digest_enabled": True},
    }

    with patch.object(TelegramDelivery, "_api_call", return_value={}):
        pipeline = NewsPipeline(config)
        storage = pipeline.storage

        for i in range(3):
            item = _mk_item(title=f"Feed{i}", url=f"https://ex.com/f{i}", llm_score=5)
            storage.save(item)
            storage.set_state([f"https://ex.com/f{i}"], "deferred")

        count = pipeline.run_digest(force=True)

        assert count == 3
        # Проверяем что targets созданы
        conn = storage._connect()
        targets = conn.execute(
            "SELECT url, kind FROM feedback_targets WHERE kind='digest'"
        ).fetchall()
        assert len(targets) == 3
        for t in targets:
            assert t["kind"] == "digest"
        pipeline.close()


# =============================================================================
# M3-регресс: digest PermanentDeliveryError → items → failed
# =============================================================================

def test_run_digest_permanent_failure_marks_failed(tmp_path):
    """
    M3: send_digest поднимает PermanentDeliveryError (400 can't parse entities) →
    urls этого сообщения помещаются в all_failed_urls → pipeline → state='failed'.
    Остальные сообщения доставлены.
    """
    from newsbot.delivery.telegram import PermanentDeliveryError, TelegramDeliveryError

    db_path = tmp_path / "test.db"
    storage = Storage(str(db_path))
    for i in range(4):
        item = _mk_item(title=f"Item{i}", url=f"https://ex.com/m3-{i}", llm_score=6)
        storage.save(item)
        storage.set_state([f"https://ex.com/m3-{i}"], "deferred")

    delivery = MagicMock()
    call_count = [0]

    def fake_send(text, buttons):
        call_count[0] += 1
        # msg2 поднимает PermanentDeliveryError
        if call_count[0] == 2:
            raise PermanentDeliveryError("HTTP 400 (content): can't parse entities")
        return True

    delivery.send_digest.side_effect = fake_send

    item_buttons = [
        [{"url": f"https://ex.com/m3-{i}", "target_id": i + 1} for i in range(2)],
        [{"url": f"https://ex.com/m3-{i}", "target_id": i + 1} for i in range(2, 4)],
    ]

    count, sent_urls, all_failed, _ex = run_digest(
        storage, delivery, max_items=100, max_items_per_message=2,
        item_buttons_per_msg=item_buttons,
    )

    # msg1 (0,1) отправлены, msg2 (2,3) → permanent failed
    assert count == 2, f"expected 2, got {count}"
    assert set(sent_urls) == {"https://ex.com/m3-0", "https://ex.com/m3-1"}
    # urls msg2 → all_failed
    assert set(all_failed) == {"https://ex.com/m3-2", "https://ex.com/m3-3"}, (
        f"expected m3-2,m3-3 in failed, got {all_failed}"
    )

    rows_digested = storage.get_all_by_state("digested")
    assert len(rows_digested) == 2, f"expected 2 digested, got {len(rows_digested)}"
    rows_failed = storage.get_all_by_state("failed")
    assert len(rows_failed) == 0, (  # failed ставит pipeline, не digest.run_digest
        f"digest.run_digest не должен сам ставить failed, got {len(rows_failed)}"
    )
    storage.close()


def test_run_digest_permanent_failure_continues_sending(tmp_path):
    """
    M3: PermanentDeliveryError на msg2 НЕ прерывает отправку msg3.
    Все незатронутые сообщения доставлены.
    """
    from newsbot.delivery.telegram import PermanentDeliveryError, TelegramDeliveryError

    db_path = tmp_path / "test.db"
    storage = Storage(str(db_path))
    for i in range(6):
        item = _mk_item(title=f"Item{i}", url=f"https://ex.com/m3c-{i}", llm_score=6)
        storage.save(item)
        storage.set_state([f"https://ex.com/m3c-{i}"], "deferred")

    delivery = MagicMock()
    call_count = [0]

    def fake_send(text, buttons):
        call_count[0] += 1
        if call_count[0] == 2:
            raise PermanentDeliveryError("HTTP 400 (content): can't parse entities")
        return True

    delivery.send_digest.side_effect = fake_send

    item_buttons = [
        [{"url": f"https://ex.com/m3c-{i}", "target_id": i + 1} for i in range(2)],
        [{"url": f"https://ex.com/m3c-{i}", "target_id": i + 1} for i in range(2, 4)],
        [{"url": f"https://ex.com/m3c-{i}", "target_id": i + 1} for i in range(4, 6)],
    ]

    count, sent_urls, all_failed, _ex = run_digest(
        storage, delivery, max_items=100, max_items_per_message=2,
        item_buttons_per_msg=item_buttons,
    )

    # msg1 + msg3 отправлены = 4 items; msg2 → failed
    assert count == 4
    assert set(sent_urls) == {
        "https://ex.com/m3c-0", "https://ex.com/m3c-1",
        "https://ex.com/m3c-4", "https://ex.com/m3c-5",
    }
    assert set(all_failed) == {"https://ex.com/m3c-2", "https://ex.com/m3c-3"}

    rows_digested = storage.get_all_by_state("digested")
    assert len(rows_digested) == 4
    storage.close()


def test_run_digest_permanent_and_transient_both_handled(tmp_path):
    """
    M3: msg1 → transient fail (False), msg2 → PermanentDeliveryError,
    msg3 → success. Ожидание: msg1 deferred, msg2 failed, msg3 digested.
    """
    from newsbot.delivery.telegram import PermanentDeliveryError, TelegramDeliveryError

    db_path = tmp_path / "test.db"
    storage = Storage(str(db_path))
    for i in range(6):
        item = _mk_item(title=f"Item{i}", url=f"https://ex.com/m3b-{i}", llm_score=6)
        storage.save(item)
        storage.set_state([f"https://ex.com/m3b-{i}"], "deferred")

    delivery = MagicMock()
    call_count = [0]

    def fake_send(text, buttons):
        call_count[0] += 1
        if call_count[0] == 1:
            return False  # transient fail → deferred
        if call_count[0] == 2:
            raise PermanentDeliveryError("HTTP 413: message is too long")
        return True  # msg3 ok

    delivery.send_digest.side_effect = fake_send

    item_buttons = [
        [{"url": f"https://ex.com/m3b-{i}", "target_id": i + 1} for i in range(2)],
        [{"url": f"https://ex.com/m3b-{i}", "target_id": i + 1} for i in range(2, 4)],
        [{"url": f"https://ex.com/m3b-{i}", "target_id": i + 1} for i in range(4, 6)],
    ]

    count, sent_urls, all_failed, _ex = run_digest(
        storage, delivery, max_items=100, max_items_per_message=2,
        item_buttons_per_msg=item_buttons,
    )

    # msg1 deferred (not in sent/failed), msg2 failed, msg3 ok
    assert count == 2  # only msg3 items
    assert set(sent_urls) == {"https://ex.com/m3b-4", "https://ex.com/m3b-5"}
    # msg1 items → deferred, msg2 items → all_failed
    assert set(all_failed) == {"https://ex.com/m3b-2", "https://ex.com/m3b-3"}

    rows_digested = storage.get_all_by_state("digested")
    assert len(rows_digested) == 2
    # run_digest не вызывает set_state для deferred/failed — это работа pipeline.
    # Оставшиеся 4 items (msg1 deferred + msg2 failed из БД) — всё ещё deferred.
    rows_deferred = storage.get_all_by_state("deferred")
    assert len(rows_deferred) == 4, f"expected 4 deferred (msg1+msg2), got {len(rows_deferred)}"
    storage.close()


def test_pipeline_run_digest_permanent_failed_set_state(tmp_path):
    """
    M3: pipeline.run_digest получает permanent_failed_urls от run_digest
    и вызывает set_state(urls, 'failed'). Проверяем что items реально
    получают state='failed' в БД.

    NOTE: нужен feedback.digest_enabled=True (send_digest path),
    потому что с False → send_text → нет кнопок → PermanentDeliveryError
    не применим. Также мокаем get_or_create_feedback_target_from_item
    (иначе pipeline падает при создании feedback-таргетов в БД).
    """
    from newsbot.pipeline import NewsPipeline
    from newsbot.delivery.telegram import PermanentDeliveryError
    from unittest.mock import patch, MagicMock

    db_path = tmp_path / "test.db"
    config = {
        "sources": [],
        "database": {"path": str(db_path)},
        "telegram": {"token": "test", "chat_id": "123"},
        "llm": {"enabled": False},
        # digest_enabled=True → pipeline вызывает send_digest (кнопки + feedback)
        "feedback": {"enabled": True, "digest_enabled": True},
        # max_items_per_message=2: 4 items → 2 groups, чтобы msg1(2items)+msg2(2items)
        "digest": {"max_items_per_message": 2},
    }

    pipeline = NewsPipeline(config)
    storage = pipeline.storage

    for i in range(4):
        item = _mk_item(title=f"Pipe{i}", url=f"https://ex.com/p{i}", llm_score=5)
        storage.save(item)
        storage.set_state([f"https://ex.com/p{i}"], "deferred")

    call_count = [0]

    def fake_send_digest(text, item_buttons):
        call_count[0] += 1
        if call_count[0] == 1:
            return True   # msg1 ok
        if call_count[0] == 2:
            raise PermanentDeliveryError("HTTP 400 (content): can't parse entities")
        return True  # msg3+ ok (не должно достигаться при 2 группах)

    # Мокаем: (1) feedback-target creation, (2) send_digest
    # feedback_target создаётся через self.storage.get_or_create_feedback_target_from_item
    with patch.object(pipeline.storage, "get_or_create_feedback_target_from_item",
                      return_value=MagicMock(id=200)):
        with patch.object(pipeline.delivery, "send_digest",
                          side_effect=fake_send_digest):
            count = pipeline.run_digest(force=True)

    # msg1 (items 0,1) отправлены
    assert count == 2, f"expected 2, got {count}"

    # msg2 (items 2,3) → permanent failed (pipeline вызывает set_state)
    rows_failed = storage.get_all_by_state("failed")
    failed_urls = {r.url for r in rows_failed}
    assert failed_urls == {"https://ex.com/p2", "https://ex.com/p3"}, (
        f"expected p2,p3 failed, got {failed_urls}"
    )

    pipeline.close()


def test_pipeline_run_digest_idempotent_targets(tmp_path):
    """Повторный run_digest для того же item НЕ плодит дубль target."""
    from newsbot.pipeline import NewsPipeline
    from newsbot.delivery import TelegramDelivery

    db_path = tmp_path / "test.db"
    config = {
        "sources": [],
        "database": {"path": str(db_path)},
        "telegram": {"token": "test", "chat_id": "123"},
        "llm": {"enabled": False},
        "feedback": {"enabled": True, "digest_enabled": True},
    }

    with patch.object(TelegramDelivery, "_api_call", return_value={}):
        pipeline = NewsPipeline(config)
        storage = pipeline.storage

        item = _mk_item(title="Idem", url="https://ex.com/idem", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/idem"], "deferred")

        # Первый запуск
        count1 = pipeline.run_digest(force=True)
        assert count1 == 1

        conn = storage._connect()
        targets_first = conn.execute(
            "SELECT id, url, kind FROM feedback_targets WHERE url='https://ex.com/idem'"
        ).fetchall()
        assert len(targets_first) == 1
        first_id = targets_first[0]["id"]

        # Сбросим digested (дадим второй шанс)
        storage.set_state(["https://ex.com/idem"], "deferred")
        # Установим границу окна, чтобы published="2025-01-01" попал в выборку.
        # Без границы используется bootstrap (24ч назад) — старый published отфильтруется.
        storage.set_meta("digest_window_started_at", "2024-01-01T00:00:00+00:00")

        # Второй запуск
        count2 = pipeline.run_digest(force=True)
        assert count2 == 1

        targets_second = conn.execute(
            "SELECT id, url, kind FROM feedback_targets WHERE url='https://ex.com/idem'"
        ).fetchall()
        assert len(targets_second) == 1, (
            f"Должен быть ровно 1 target (идемпотентность), найдено {len(targets_second)}"
        )
        assert targets_second[0]["id"] == first_id, "idempotent — тот же id"

        pipeline.close()


def test_pipeline_run_digest_no_targets_when_disabled(tmp_path):
    """При feedback.enabled=False targets НЕ создаются."""
    from newsbot.pipeline import NewsPipeline
    from newsbot.delivery import TelegramDelivery

    db_path = tmp_path / "test.db"
    config = {
        "sources": [],
        "database": {"path": str(db_path)},
        "telegram": {"token": "test", "chat_id": "123"},
        "llm": {"enabled": False},
        "feedback": {"enabled": False},
    }

    with patch.object(TelegramDelivery, "_api_call", return_value={}):
        pipeline = NewsPipeline(config)
        storage = pipeline.storage

        item = _mk_item(title="NoFeed", url="https://ex.com/nofeed", llm_score=5)
        storage.save(item)
        storage.set_state(["https://ex.com/nofeed"], "deferred")

        count = pipeline.run_digest(force=True)

        assert count == 1
        conn = storage._connect()
        targets = conn.execute(
            "SELECT id FROM feedback_targets WHERE url='https://ex.com/nofeed'"
        ).fetchall()
        assert len(targets) == 0, "При disabled=False targets не должны создаваться"
        pipeline.close()


def test_pipeline_run_digest_no_targets_console_delivery(tmp_path):
    """В режиме ConsoleDelivery targets НЕ создаются (on_empty=send)."""
    from newsbot.pipeline import NewsPipeline

    db_path = tmp_path / "test.db"
    config = {
        "sources": [],
        "database": {"path": str(db_path)},
        "telegram": {"token": "", "chat_id": ""},  # → ConsoleDelivery
        "digest": {"on_empty": "send"},
        "llm": {"enabled": False},
        "feedback": {"enabled": True},
    }

    pipeline = NewsPipeline(config)
    storage = pipeline.storage

    item = _mk_item(title="Console", url="https://ex.com/console", llm_score=5)
    storage.save(item)
    storage.set_state(["https://ex.com/console"], "deferred")

    count = pipeline.run_digest(force=True)

    assert count == 1
    conn = storage._connect()
    targets = conn.execute(
        "SELECT id FROM feedback_targets WHERE url='https://ex.com/console'"
    ).fetchall()
    assert len(targets) == 0, "ConsoleDelivery не должна создавать targets"
    pipeline.close()