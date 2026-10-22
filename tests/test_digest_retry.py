"""
tests/test_digest_retry.py — тесты фиксов дайджеста.

Покрывает:
  1. M2: пустой deferred пишет last_digest_at (окно проверено)
  2. Minor (URL>лимита): футер без <a>, а не битая ссылка
  3. Minor (deferred terminal limit): счётчик + терминализация после лимита

Запуск: python -m pytest tests/test_digest_retry.py -v
"""
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from newsbot.storage import Storage
from newsbot.collector import NewsItem
from newsbot.digest import build_digest_messages, run_digest
from newsbot.pipeline import NewsPipeline


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
# M2: пустой deferred пишет last_digest_at (run_digest не спамит в след.цикле)
# =============================================================================

def test_pipeline_run_digest_empty_deferred_writes_last_digest_at(tmp_path):
    """
    M2: пустой deferred + on_empty=skip → last_digest_at пишется.
    Без фикса: early return пропускал запись last_digest_at, _maybe_run_digest
    запускал run_digest КАЖДЫЙ цикл (лог-спам, первый deferred уходил по старому окну).
    """
    db_path = tmp_path / "test.db"
    config = {
        "sources": [],
        "database": {"path": str(db_path)},
        "telegram": {"token": "", "chat_id": ""},
        "digest": {"enabled": True, "on_empty": "skip",
                   "max_items": 100, "max_items_per_message": 20},
        "llm": {"enabled": False},
    }

    pipeline = NewsPipeline(config)
    delivery = MagicMock()
    pipeline.delivery = delivery

    # Нет deferred items
    count = pipeline.run_digest(force=False)

    assert count == 0
    # M2-фикс: last_digest_at записан
    val = pipeline.storage.get_meta("last_digest_at")
    assert val is not None, "last_digest_at должен быть записан при пустом deferred"
    # Проверяем что это валидный isoformat
    dt = datetime.fromisoformat(val)
    assert dt.year >= 2024
    # send_text НЕ вызывался (on_empty=skip)
    delivery.send_text.assert_not_called()
    pipeline.close()


def test_pipeline_run_digest_empty_deferred_on_send_no_notification(tmp_path):
    """
    M2: on_empty=send + пустой deferred → send_text вызывается,
    last_digest_at пишется (проверено предыдущим тестом).
    """
    db_path = tmp_path / "test.db"
    config = {
        "sources": [],
        "database": {"path": str(db_path)},
        "telegram": {"token": "", "chat_id": ""},
        "digest": {"enabled": True, "on_empty": "send",
                   "max_items": 100, "max_items_per_message": 20},
        "llm": {"enabled": False},
    }

    pipeline = NewsPipeline(config)
    delivery = MagicMock()
    pipeline.delivery = delivery

    count = pipeline.run_digest(force=False)

    assert count == 0
    # on_empty=send: уведомление отправлено
    delivery.send_text.assert_called_once()
    val = pipeline.storage.get_meta("last_digest_at")
    assert val is not None
    pipeline.close()


def test_maybe_run_digest_no_spam_after_empty_deferred(tmp_path):
    """
    M2: после пустого deferred (_maybe_run_digest отработал) — второй вызов
    НЕ запускает digest повторно (last_digest_at блокирует).
    """
    db_path = tmp_path / "test.db"
    item = _mk_item(title="X", url="https://ex.com/x", llm_score=5)
    config = {
        "sources": [],
        "database": {"path": str(db_path)},
        "telegram": {"token": "", "chat_id": ""},
        "digest": {"enabled": True, "on_empty": "skip",
                   "max_items": 100, "max_items_per_message": 20},
        "llm": {"enabled": False},
    }

    pipeline = NewsPipeline(config)
    pipeline.storage.save(item)
    pipeline.storage.set_state(["https://ex.com/x"], "deferred")

    delivery = MagicMock()
    pipeline.delivery = delivery

    now_local = datetime.now(timezone.utc).astimezone()
    current_hhmm = now_local.strftime("%H:%M")
    pipeline.config["digest"]["times"] = [current_hhmm]

    # Первый вызов: отправляем дайджест (есть deferred)
    pipeline._maybe_run_digest()
    assert delivery.send_text.called
    delivery.reset_mock()

    # Второй вызов: last_digest_at обновлён, окно обработано — skip
    pipeline._maybe_run_digest()
    delivery.send_text.assert_not_called()

    pipeline.close()


# =============================================================================
# Minor (URL>лимита): футер БЕЗ <a>, не битая ссылка
# =============================================================================

def test_long_url_no_link_in_footer(tmp_path):
    """
    Minor: URL с escaped>500 символов → footer БЕЗ <a href>, только время/текст.
    Без фикса: URL обрезался prefix + подставлялся в href → битая ссылка.
    """
    # URL с 500 амперсандами → escaped ~1500 символов (3× раздутие)
    long_url = "https://ex.com/?" + "&" * 500 + "q=1"
    item = _mk_item(
        title="Long URL",
        url=long_url,
        summary="Short.",
        llm_score=5,
    )
    groups, _, _ = build_digest_messages([item], max_items_per_message=1)
    assert len(groups) == 1
    msg, _ = groups[0]

    # Нет <a href> (битая ссылка)
    assert '<a href="' not in msg, "Не должно быть <a href> для слишком длинного URL"
    # Время/текст присутствует
    assert "🔗" not in msg, "Иконка ссылки не должна появляться при отсутствии <a>"
    # HTML валиден
    assert msg.count("<b>") == msg.count("</b>")


def test_run_digest_long_url_no_broken_link(tmp_path):
    """
    Minor: end-to-end с слишком длинным URL — футер без <a>, не битая ссылка.
    """
    db_path = tmp_path / "test.db"
    long_url = "https://ex.com/?" + "&" * 500 + "q=1"
    item = _mk_item(title="X", url=long_url, summary="Short.", llm_score=5)

    config = {
        "sources": [],
        "database": {"path": str(db_path)},
        "telegram": {"token": "", "chat_id": ""},
        "digest": {"enabled": True, "on_empty": "skip",
                   "max_items": 100, "max_items_per_message": 20},
        "llm": {"enabled": False},
    }

    pipeline = NewsPipeline(config)
    pipeline.storage.save(item)
    pipeline.storage.set_state([long_url], "deferred")

    delivery = MagicMock()
    delivery.send_text.return_value = True
    pipeline.delivery = delivery

    count = pipeline.run_digest(force=True)

    assert count == 1
    # Текст в send_text — без <a href>
    args = delivery.send_text.call_args[0][0]
    assert '<a href="' not in args, "Не должно быть <a href> в отправленном сообщении"
    pipeline.close()


# =============================================================================
# Critical-фикс: deferred НЕ терминализируются по неуспешным попыткам дайджеста.
# retry_count НЕ инкрементируется, НЕ переводится в 'failed'.
# Deferred остаются для повтора в следующем окне (at-least-once).
# =============================================================================

def test_digest_transient_failure_deferred_not_failed(tmp_path):
    """
    Critical: полный транзиентный сбой дайджеста — deferred НЕ помечаются failed.
    При следующем окне они повторяются.
    """
    import sqlite3

    db_path = tmp_path / "test.db"
    items = [
        _mk_item(title=f"D{i}", url=f"https://ex.com/d{i}", llm_score=5)
        for i in range(3)
    ]

    config = {
        "sources": [],
        "database": {"path": str(db_path), "max_retries": 3},
        "telegram": {"token": "", "chat_id": ""},
        "digest": {"enabled": True, "on_empty": "skip",
                   "max_items": 100, "max_items_per_message": 20},
        "llm": {"enabled": False},
    }

    pipeline = NewsPipeline(config)
    for item in items:
        pipeline.storage.save(item)
        pipeline.storage.set_state([item.url], "deferred")

    delivery = MagicMock()
    delivery.send_text.return_value = False  # полный сбой
    pipeline.delivery = delivery

    # Первый вызов: сбой
    count = pipeline.run_digest(force=True)
    assert count == 0

    # Deferred остаются deferred (НЕ failed)
    rows_failed = pipeline.storage.get_all_by_state("failed")
    assert len(rows_failed) == 0, "Deferred не должны становиться failed при сбое дайджеста"

    rows_deferred = pipeline.storage.get_all_by_state("deferred")
    assert len(rows_deferred) == 3, f"Все 3 deferred должны остаться, got {len(rows_deferred)}"

    # retry_count не инкрементирован
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT url, retry_count FROM items WHERE url LIKE 'https://ex.com/d%'"
    ).fetchall()
    conn.close()
    for url, retry_count in rows:
        assert retry_count == 0, f"{url}: retry_count должен быть 0, got {retry_count}"

    # last_digest_at НЕ записан (транзиентный сбой)
    val = pipeline.storage.get_meta("last_digest_at")
    assert val is None, "last_digest_at не должен писаться при сбое"

    pipeline.close()


def test_digest_transient_failure_repeated_in_next_window(tmp_path):
    """
    Critical: после транзиентного сбоя deferred повторяются в следующем окне.
    """
    import sqlite3
    from datetime import datetime, timezone

    db_path = tmp_path / "test.db"

    # Инициализируем схему БД (Storage создаёт таблицы).
    # Вставляем напрямую с явным fetched_at в ПРОШЛОМ.
    # Storage.save() ставит NOW, что может совпасть с digest_started_iso
    # первого вызова и сломать детерминированность (item не удалится).
    # Явный fetched_at в 2025 г. гарантирует удаление при cleanup.
    _ = Storage(str(db_path))  # создаёт схему items

    fetched_at = "2025-01-01T12:00:00+00:00"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT OR IGNORE INTO items
        (source, url, title, summary, published, fetched_at, state, state_at, llm_score)
        VALUES (?, ?, ?, ?, ?, ?, 'deferred', ?, ?)
        """,
        ("TestSrc", "https://ex.com/willretry", "WillRetry",
         "Test summary", "2025-01-01 12:00", fetched_at, fetched_at, 5),
    )
    conn.commit()
    conn.close()

    config = {
        "sources": [],
        "database": {"path": str(db_path)},
        "telegram": {"token": "", "chat_id": ""},
        "digest": {"enabled": True, "on_empty": "skip",
                   "max_items": 100, "max_items_per_message": 20},
        "llm": {"enabled": False},
    }

    pipeline = NewsPipeline(config)

    delivery = MagicMock()
    call_num = [0]

    def fake_send(text):
        call_num[0] += 1
        return call_num[0] >= 2  # второй вызов — успех

    delivery.send_text.side_effect = fake_send
    pipeline.delivery = delivery

    # Первый вызов: сбой
    count1 = pipeline.run_digest(force=True)
    assert count1 == 0

    # Второй вызов (имитация следующего окна): успех
    count2 = pipeline.run_digest(force=True)
    assert count2 == 1, f"Ожидался 1 отправленный item, got {count2}"

    # После успешного дайджеста ВСЁ окно удалено из БД.
    # Item отправлен, но удалён delete_items_before (fetched_at в прошлом).
    all_urls = set()
    for state in ("deferred", "digested", "new", "pending", "skipped", "failed"):
        all_urls.update(i.url for i in pipeline.storage.get_all_by_state(state))
    assert "https://ex.com/willretry" not in all_urls, (
        "Item должен быть удалён из БД после успешного дайджеста (окно очищено)"
    )
    pipeline.close()


def test_digest_partial_failure_deferred_persists(tmp_path):
    """
    Critical: частичный сбой — отправленные → digested, неотправленные остаются deferred.
    retry_count не трогается.
    """
    import sqlite3

    db_path = tmp_path / "test.db"
    items = [
        _mk_item(title=f"Item{i}", url=f"https://ex.com/item{i}", llm_score=6)
        for i in range(3)
    ]

    config = {
        "sources": [],
        "database": {"path": str(db_path)},
        "telegram": {"token": "", "chat_id": ""},
        "digest": {"enabled": True, "on_empty": "skip",
                   "max_items": 100, "max_items_per_message": 1},
        "llm": {"enabled": False},
    }

    pipeline = NewsPipeline(config)
    for item in items:
        pipeline.storage.save(item)
        pipeline.storage.set_state([item.url], "deferred")

    delivery = MagicMock()
    call_count = [0]

    def fake_send(text):
        call_count[0] += 1
        return call_count[0] == 1  # только первое сообщение ok

    delivery.send_text.side_effect = fake_send
    pipeline.delivery = delivery

    count = pipeline.run_digest(force=True)

    # item0 отправлен (ok), item1+item2 не отправлены (fail)
    assert count == 1

    # Отправленные → digested
    rows_digested = pipeline.storage.get_all_by_state("digested")
    assert len(rows_digested) == 1

    # Неотправленные остаются deferred (НЕ failed, retry_count не трогается)
    rows_failed = pipeline.storage.get_all_by_state("failed")
    assert len(rows_failed) == 0, "Неотправленные deferred НЕ должны становиться failed"

    rows_deferred = pipeline.storage.get_all_by_state("deferred")
    assert len(rows_deferred) == 2, f"Неотправленные остаются deferred, got {len(rows_deferred)}"

    # retry_count всех deferred остаётся 0
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT url, retry_count FROM items WHERE url LIKE 'https://ex.com/item%' AND state='deferred'"
    ).fetchall()
    conn.close()
    for url, retry_count in rows:
        assert retry_count == 0, f"{url}: retry_count должен быть 0, got {retry_count}"

    pipeline.close()


def test_digest_deferred_retry_after_many_cycles(tmp_path):
    """
    Critical: deferred не терминализируются после max_retries попыток дайджеста.
    Они остаются deferred до успешной отправки или retention-purge.
    """
    import sqlite3

    db_path = tmp_path / "test.db"
    item = _mk_item(title="Persist", url="https://ex.com/persist", llm_score=5)

    config = {
        "sources": [],
        "database": {"path": str(db_path), "max_retries": 3},
        "telegram": {"token": "", "chat_id": ""},
        "digest": {"enabled": True, "on_empty": "skip",
                   "max_items": 100, "max_items_per_message": 20},
        "llm": {"enabled": False},
    }

    pipeline = NewsPipeline(config)
    pipeline.storage.save(item)
    pipeline.storage.set_state(["https://ex.com/persist"], "deferred")

    # Предварительно выставляем retry_count = 10 (>> max_retries=3)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "UPDATE items SET retry_count=10 WHERE url='https://ex.com/persist'"
    )
    conn.commit()
    conn.close()

    delivery = MagicMock()
    delivery.send_text.return_value = False  # сбой
    pipeline.delivery = delivery

    count = pipeline.run_digest(force=True)

    assert count == 0

    # Несмотря на retry_count >> max_retries, deferred остаётся deferred
    rows_failed = pipeline.storage.get_all_by_state("failed")
    assert len(rows_failed) == 0, "Deferred с большим retry_count НЕ должен становиться failed"

    rows_deferred = pipeline.storage.get_all_by_state("deferred")
    assert len(rows_deferred) == 1, f"Deferred должен остаться, got {len(rows_deferred)}"
    pipeline.close()


# =============================================================================
# Major-фикс: scored-stuck доставляются при _llm_calls_left==0
# =============================================================================

def test_scored_stuck_delivered_when_llm_budget_exhausted(tmp_path):
    """
    Major: scored-items (llm_score сохранён) доставляются при _llm_calls_left==0
    благодаря retry_batch (не тратят бюджет), тогда как fresh-items остаются pending.
    """
    from unittest.mock import PropertyMock
    from newsbot.collector import NewsItem
    from newsbot.llm.filter import FilterResult

    db_path = tmp_path / "test.db"
    # Scored item (llm_score уже сохранён) — retry_batch, не тратит бюджет
    scored_item = NewsItem(
        source="Src", url="https://ex.com/scored1",
        title="Scored Title", summary="Summary",
        published="2025-01-01 12:00", published_parsed=None,
        llm_score=7, llm_summary="LLM summary", llm_reason="good",
        llm_urgent=True,
    )
    # Fresh item (llm_score=None) — fresh_batch, ТРАТИТ бюджет
    fresh_item = NewsItem(
        source="Src2", url="https://ex.com/fresh1",
        title="Fresh Title", summary="Summary2",
        published="2025-01-01 12:00", published_parsed=None,
        llm_score=None,
    )

    config = {
        "sources": [],
        "database": {"path": str(db_path)},
        "telegram": {"token": "", "chat_id": ""},
        "llm": {
            "enabled": True,
            "batch_size": 10,
            "push_threshold": 7,
            "digest_threshold": 5,
        },
        "digest": {"enabled": False},
    }

    pipeline = NewsPipeline(config)
    pipeline.storage.save(scored_item)
    pipeline.storage.save(fresh_item)
    pipeline.storage.set_state(["https://ex.com/scored1"], "pending")
    pipeline.storage.set_state(["https://ex.com/fresh1"], "pending")

    # Бюджет = 1: хватит только scored-item (retry, не тратит бюджет).
    # Fresh-item будет срезан бюджетом, но НЕ попадает в failed_urls.
    # Мок-фильтр: пусть fresh-item тоже прошёл бы, но он не дойдёт до LLM
    # (budget_override=1 в _process_batch управляет бюджетом напрямую,
    # _llm_calls_left не читается в этом пути).
    mock_llm = MagicMock()
    mock_llm.push_threshold = 7
    mock_llm.digest_threshold = 5
    mock_llm.push_urgent_only = False
    mock_llm.evaluate.side_effect = lambda item, feedback_context=None: FilterResult(
        ok=True, score=10, summary="s", reason="r"
    )
    mock_llm.evaluate_batch.side_effect = lambda items, feedback_context=None: [
        mock_llm.evaluate(item, feedback_context) for item in items
    ]
    pipeline.llm_filter = mock_llm

    delivery = MagicMock()
    delivery.send_batch.return_value = ["https://ex.com/scored1"]
    pipeline.delivery = delivery

    # batch_size=10, бюджет=1, 2 items → обработан только scored (retry)
    failed = pipeline._process_batch([scored_item, fresh_item], budget_override=1)

    # Scored доставлен
    assert delivery.send_batch.called, "Scored item должен быть отправлен"
    # Fresh срезан бюджетом — НЕ в failed_urls (budget cut ≠ delivery failure)
    assert "https://ex.com/fresh1" not in failed, (
        "Fresh item срезанный бюджетом НЕ должен быть в failed_urls"
    )
    # Fresh остался pending
    rows = pipeline.storage.get_all_by_state("pending")
    pending_urls = [r.url for r in rows]
    assert "https://ex.com/fresh1" in pending_urls, (
        "Fresh item должен остаться pending (бюджет исчерпан)"
    )
    # LLM-вызов по scored НЕ должен был состояться (retry_batch — synthesize)
    assert not mock_llm.evaluate_batch.called, (
        "evaluate_batch НЕ должен вызываться для scored items"
    )
    pipeline.close()


# =============================================================================
# Minor (zip sync): рассинхрон groups vs buttons — логируется, не теряется
# =============================================================================

def test_send_digest_mismatch_logs_warning(tmp_path, caplog):
    """Minor: zip mismatch groups vs item_buttons_per_msg логируется с корректным текстом."""
    import logging

    db_path = tmp_path / "test.db"
    storage = Storage(str(db_path))
    for i in range(4):
        item = _mk_item(title=f"X{i}", url=f"https://ex.com/x{i}", llm_score=5)
        storage.save(item)
        storage.set_state([f"https://ex.com/x{i}"], "deferred")

    delivery = MagicMock()
    delivery.send_digest.return_value = True

    # 4 items → 2 groups (max=2), но передаём 5 кнопок
    item_buttons = [[{"t": 1}] for _ in range(5)]
    groups, _, _ = build_digest_messages(
        [storage.get_deferred_items()[0],
         storage.get_deferred_items()[1] or _mk_item(title="X0b", url="https://ex.com/x0b", llm_score=5)],
        max_items_per_message=2,
    )
    # Простой случай: buttons длиннее groups
    with caplog.at_level(logging.WARNING, logger="newsbot.digest"):
        count, urls, _failed = run_digest(
            storage, delivery, max_items=100, max_items_per_message=20,
            item_buttons_per_msg=item_buttons,
            groups=groups,
        )

    assert count == 2
    # WARNING должен был залогироваться (buttons длиннее)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "рассинхрон" in warnings[0].message
    storage.close()