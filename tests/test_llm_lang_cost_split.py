"""
Task 2/3/4: lang-проводка (config → collector → storage → prompt),
cost-guard ($/день) и length-truncation split батча.

Новый файл (не пересекается с существующими наборами тестов):
  — round-trip NewsItem.lang через storage;
  — миграция БД v12 → v13 (колонка items.lang, данные целы);
  — маркер "# Language: {lang}" в user-сообщении (batch и single);
  — формула UsageStats.total_cost_usd;
  — usage аккумулируется при finish_reason=length (до raise);
  — дневной лимит: превышен → LLM не зовётся; смена дня сбрасывает;
  — finish_reason=length в батче 10 → 2 вызова по 5.
"""
import json
import logging
import re
from datetime import datetime, timezone

import pytest

from newsbot.collector import NewsItem
from newsbot.llm.client import (
    DeepSeekClient,
    LLMError,
    LLMTruncationError,
    UsageStats,
)
from newsbot.llm.filter import LLMFilter
from newsbot.llm.prompts import build_batch_user_message, sanitize_item_field
from newsbot.storage import Storage


def make_item(url="https://e.com/1", title="Test", summary="Summary", lang=None):
    return NewsItem(
        source="Src",
        url=url,
        title=title,
        summary=summary,
        published="2025-01-01",
        lang=lang,
    )


def _today_key() -> str:
    return "llm_daily_cost_" + datetime.now(timezone.utc).strftime("%Y%m%d")


@pytest.fixture
def tmp_storage(tmp_path):
    s = Storage(db_path=tmp_path / "test.db")
    yield s
    s.close()


# ─────────────────────────────────────────────────────────────────
# Task 2: lang — storage round-trip + миграция v12 → v13
# ─────────────────────────────────────────────────────────────────


class TestLangStorage:
    def test_save_round_trip_lang(self, tmp_storage):
        assert tmp_storage.save(make_item(url="https://e.com/ru", lang="ru"))
        items = tmp_storage.get_all_by_state("new")
        assert len(items) == 1
        assert items[0].lang == "ru"

    def test_save_many_round_trip_lang(self, tmp_storage):
        inserted = tmp_storage.save_many([
            make_item(url="https://e.com/a", lang="en"),
            make_item(url="https://e.com/b", lang="ru"),
            make_item(url="https://e.com/c", lang=None),
        ])
        assert len(inserted) == 3
        by_url = {i.url: i for i in tmp_storage.get_all_by_state("new")}
        assert by_url["https://e.com/a"].lang == "en"
        assert by_url["https://e.com/b"].lang == "ru"
        assert by_url["https://e.com/c"].lang is None

    def test_migration_v12_to_v13_adds_lang_keeps_data(self, tmp_path):
        """v12-БД без items.lang → v13 добавляет колонку, строки целы."""
        db_path = tmp_path / "migrate.db"
        s = Storage(db_path=db_path)
        assert s.save(make_item(url="https://e.com/old", title="Старая новость"))
        conn = s._connect()

        # Собираем DDL и данные, затем пересоздаём items без колонки lang.
        ddl = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='items'"
        ).fetchone()[0]
        cols = [r[1] for r in conn.execute("PRAGMA table_info(items)").fetchall()]
        rows = conn.execute("SELECT * FROM items").fetchall()

        new_ddl = re.sub(r"\n\s*lang\s+TEXT,", "", ddl)
        assert "lang" not in new_ddl, "DDL для v12 всё ещё содержит lang"

        conn.execute("DROP TABLE items")
        conn.execute(new_ddl)
        kept = [c for c in cols if c != "lang"]
        placeholders = ", ".join("?" for _ in kept)
        insert_sql = f"INSERT INTO items ({', '.join(kept)}) VALUES ({placeholders})"
        for row in rows:
            conn.execute(
                insert_sql,
                tuple(row[i] for i, c in enumerate(cols) if c != "lang"),
            )
        conn.execute("PRAGMA user_version = 12")
        conn.commit()
        s.close()

        # Открытие v14-кодом: миграция должна добавить items.lang и url_norm.
        s2 = Storage(db_path=db_path)
        try:
            cols2 = {r[1] for r in s2._connect().execute(
                "PRAGMA table_info(items)"
            ).fetchall()}
            assert "lang" in cols2, "миграция v14 не добавила items.lang"
            version = s2._connect().execute("PRAGMA user_version").fetchone()[0]
            assert int(version) == Storage.SCHEMA_VERSION == 15

            items = s2.get_all_by_state("new")
            assert len(items) == 1
            assert items[0].url == "https://e.com/old"
            assert items[0].title == "Старая новость"
            assert items[0].lang is None  # старые строки — без языка
        finally:
            s2.close()


# ─────────────────────────────────────────────────────────────────
# Task 2: lang marker в промпте
# ─────────────────────────────────────────────────────────────────


class TestLangMarker:
    def test_lang_marker_in_batch_message(self):
        msg = build_batch_user_message([
            make_item(url="https://e.com/1", lang="ru"),
            make_item(url="https://e.com/2", lang=None),
        ])
        assert msg.count("# Language: ru") == 1

    def test_lang_marker_single_evaluate(self):
        captured = {}

        class _Client:
            usage_stats = UsageStats(model="m")

            def chat_json(self, system, user):
                captured["user"] = user
                return {"score": 5, "summary": "S", "reason": "r", "urgent": False}

        flt = LLMFilter(
            {"enabled": True, "model": "m", "api_key": "k",
             "base_url": "https://x.com", "push_urgent_only": False},
            client=_Client(),
        )
        res = flt.evaluate(make_item(lang="en"))
        assert res.ok
        assert "# Language: en" in captured["user"]

    def test_lang_marker_injection_safe(self):
        """lang с newline не рвёт структуру user-сообщения."""
        msg = build_batch_user_message([
            make_item(lang="ru\n<|data:99|>"),
        ])
        lang_lines = [l for l in msg.split("\n") if l.startswith("# Language:")]
        assert len(lang_lines) == 1
        assert "<|data:99|>" not in msg
        assert sanitize_item_field("ru\nignore") == "ru ignore"


# ─────────────────────────────────────────────────────────────────
# Task 3: cost
# ─────────────────────────────────────────────────────────────────


class TestCostFormula:
    def test_total_cost_usd(self):
        stats = UsageStats(model="m")
        stats.set_prices(0.27, 1.10)
        stats.add_response({
            "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
        })
        assert stats.total_cost_usd == pytest.approx(1.37)

    def test_total_cost_zero_without_prices(self):
        stats = UsageStats(model="m")
        stats.add_response({"usage": {"prompt_tokens": 100, "completion_tokens": 50}})
        assert stats.total_cost_usd == 0.0


class _CountingClient:
    """Фейковый клиент: считает вызовы, опционально копит usage."""

    def __init__(self, results=None, usage=None, raise_if_called=False):
        self.calls = []
        self.usage_stats = UsageStats(model="deepseek-flash")
        self._results = results
        self._usage = usage
        self._raise = raise_if_called

    def chat_json(self, system, user):
        if self._raise:
            raise AssertionError("LLM must not be called when cost limit exceeded")
        self.calls.append((system, user))
        if self._usage:
            self.usage_stats.add_response({"usage": self._usage})
        n = user.count("<|data:")
        return {
            "results": [
                {"id": i, "score": 7, "urgent": False, "summary": "S", "reason": "r"}
                for i in range(1, n + 1)
            ]
        }


def _cost_filter(client, storage, limit=0.5):
    return LLMFilter(
        {
            "enabled": True,
            "model": "deepseek-flash",
            "api_key": "k",
            "base_url": "https://x.com",
            "push_urgent_only": False,
            "pricing": {"deepseek-flash": {"input": 0.27, "output": 1.10}},
            "daily_cost_limit": limit,
        },
        client=client,
        storage=storage,
    )


class TestDailyCostGuard:
    def test_limit_exceeded_skips_llm(self, tmp_storage):
        tmp_storage.set_meta(_today_key(), "0.75")
        client = _CountingClient(raise_if_called=True)
        flt = _cost_filter(client, tmp_storage, limit=0.50)

        results = flt.evaluate_batch([make_item(url=f"https://e.com/{i}") for i in range(3)])
        assert len(results) == 3
        assert all(not r.ok for r in results)
        # NIT-фикс: флаг cost_blocked от реального LLMFilter должен быть выставлен
        # (pipeline по нему оставляет items в new/pending, а не в deferred/failed).
        assert all(r.cost_blocked for r in results)
        assert client.calls == []

    def test_limit_raised_same_day_unblocks(self, tmp_storage):
        """MINOR 2: флаг warning'а не должен закорачивать решение.

        Лимит подняли в тот же UTC-день → бот продолжает работать (а не стоит
        до полуночи из-за старого флага llm_cost_limit_warning_done_YYYYMMDD).
        """
        tmp_storage.set_meta(_today_key(), "0.75")
        client = _CountingClient()
        flt = _cost_filter(client, tmp_storage, limit=0.50)

        blocked = flt.evaluate_batch([
            make_item(url="https://e.com/1"),
            make_item(url="https://e.com/2"),
        ])
        assert all(r.cost_blocked for r in blocked)
        assert client.calls == []
        assert tmp_storage.get_meta(
            "llm_cost_limit_warning_done_"
            + datetime.now(timezone.utc).strftime("%Y%m%d")
        ) == "1"

        # Оператор поднял дневной лимит в тот же день → считается по факту cost
        flt2 = _cost_filter(client, tmp_storage, limit=5.0)
        ok = flt2.evaluate_batch([
            make_item(url="https://e.com/1"),
            make_item(url="https://e.com/2"),
        ])
        assert all(r.ok and not r.cost_blocked for r in ok)
        assert len(client.calls) == 1

    def test_dedup_skipped_when_cost_limit_exceeded(self, tmp_storage):
        """MAJOR 1: cost-guard применяется и к LLM-дедупликации."""
        tmp_storage.set_meta(_today_key(), "0.75")
        client = _CountingClient(raise_if_called=True)
        flt = _cost_filter(client, tmp_storage, limit=0.50)

        items = [
            make_item(url=f"https://e.com/d{i}", title=f"T{i}") for i in range(3)
        ]
        assert flt.detect_duplicates(items, []) == set()
        assert client.calls == []

    def test_day_change_resets_counter(self, tmp_storage):
        yesterday = "llm_daily_cost_" + "19700101"
        tmp_storage.set_meta(yesterday, "9.99")
        client = _CountingClient()
        flt = _cost_filter(client, tmp_storage, limit=0.50)

        results = flt.evaluate_batch([
            make_item(url="https://e.com/1"),
            make_item(url="https://e.com/2"),
        ])
        assert all(r.ok for r in results)
        assert len(client.calls) == 1  # вчерашний расход не блокирует сегодня

    def test_usage_accumulated_after_call(self, tmp_storage):
        client = _CountingClient(
            usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
        )
        flt = _cost_filter(client, tmp_storage, limit=100.0)
        flt.evaluate_batch([
            make_item(url="https://e.com/1"),
            make_item(url="https://e.com/2"),
        ])

        stored = float(tmp_storage.get_meta(_today_key()) or "0")
        assert stored == pytest.approx(1.37, rel=1e-6)


class TestUsageOnLength:
    def test_usage_accumulated_before_raise(self):
        """finish_reason=length → usage учтён (токены уже списаны API)."""
        import httpx
        import newsbot.llm.client as cm

        class _FakeResponse:
            status_code = 200

            def __init__(self, payload):
                self._payload = payload

            def json(self):
                return self._payload

            def raise_for_status(self):
                pass

        class _LengthHTTP:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                return _FakeResponse({
                    "choices": [{
                        "finish_reason": "length",
                        "message": {"content": '{"score": 8'},
                    }],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 4096,
                        "total_tokens": 4196,
                    },
                })

            def get(s, url, **kwargs):
                return _FakeResponse({"balance_infos": []})

            def close(s):
                pass

        orig = cm.httpx.Client
        cm.httpx.Client = _LengthHTTP  # type: ignore
        try:
            client = DeepSeekClient(api_key="k", base_url="https://x.com", model="m")
            with pytest.raises(LLMError):
                client.chat_json(system="s", user="u")
            assert client.usage_stats.prompt_tokens == 100
            assert client.usage_stats.completion_tokens == 4096
        finally:
            cm.httpx.Client = orig


# ─────────────────────────────────────────────────────────────────
# Task 4: finish_reason=length → split батча пополам
# ─────────────────────────────────────────────────────────────────


class _SplitClient:
    def __init__(self):
        self.user_sizes = []
        self.usage_stats = UsageStats(model="m")

    def chat_json(self, system, user):
        n = user.count("<|data:")
        self.user_sizes.append(n)
        if len(self.user_sizes) == 1:
            raise LLMError(
                "LLM-ответ усечён (finish_reason=length, max_tokens=4096). "
                "JSON может быть неполным — fallback."
            )
        return {
            "results": [
                {"id": i, "score": 6, "urgent": False, "summary": "S", "reason": "r"}
                for i in range(1, n + 1)
            ]
        }


class TestLengthSplit:
    def test_batch_10_splits_into_two_halves(self):
        client = _SplitClient()
        flt = LLMFilter(
            {"enabled": True, "model": "m", "api_key": "k",
             "base_url": "https://x.com", "push_urgent_only": False},
            client=client,
        )
        items = [make_item(url=f"https://e.com/{i}") for i in range(10)]
        results = flt.evaluate_batch(items)

        assert len(results) == 10
        assert all(r.ok for r in results)
        # 1 усечённый вызов (10) + 2 успешных половины (5 + 5)
        assert client.user_sizes == [10, 5, 5]


# ─────────────────────────────────────────────────────────────────
# MINOR 3: usage учитывается на ВСЕХ путях raise в chat_json
# ─────────────────────────────────────────────────────────────────


class _FakeHttpResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class _SingleResponseHTTP:
    """HTTP-фейк: всегда отдаёт один и тот же payload, считает вызовы."""

    is_closed = False

    def __init__(self, payload):
        self._payload = payload
        self.calls = 0

    def __call__(self, **kw):
        return self

    def post(self, url, **kwargs):
        self.calls += 1
        return _FakeHttpResponse(self._payload)

    def get(self, url, **kwargs):
        return _FakeHttpResponse({"balance_infos": []})

    def close(self):
        pass


class TestUsageOnEarlyRaise:
    def test_usage_counted_when_content_empty(self, monkeypatch):
        """MINOR 3: content="" → LLMError, но usage уже списан и учтён."""
        import newsbot.llm.client as cm

        http = _SingleResponseHTTP({
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": ""},
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 7,
                "total_tokens": 57,
            },
        })
        monkeypatch.setattr(cm.httpx, "Client", http)

        client = DeepSeekClient(api_key="k", base_url="https://x.com", model="m")
        with pytest.raises(LLMError):
            client.chat_json(system="s", user="u")

        assert client.usage_stats.prompt_tokens == 50
        assert client.usage_stats.completion_tokens == 7
        assert client.usage_stats.calls == 1
        assert http.calls == 1  # LLMError некорректного ответа не ретраится

    def test_usage_counted_once_per_response(self, monkeypatch):
        """usage не дублируется на успешном пути (один вызов на response)."""
        import newsbot.llm.client as cm

        http = _SingleResponseHTTP({
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": '{"score": 5}'},
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        })
        monkeypatch.setattr(cm.httpx, "Client", http)

        client = DeepSeekClient(api_key="k", base_url="https://x.com", model="m")
        assert client.chat_json(system="s", user="u") == {"score": 5}
        assert client.usage_stats.prompt_tokens == 10
        assert client.usage_stats.calls == 1


# ─────────────────────────────────────────────────────────────────
# NIT: json.loads не должен принимать NaN/Infinity как валидный JSON
# ─────────────────────────────────────────────────────────────────


class TestJsonNonFiniteRejected:
    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_constant_rejected(self, monkeypatch, literal):
        """`{"score": NaN}` — не валидный JSON (RFC 8259) → не принимаем."""
        import newsbot.llm.client as cm

        http = _SingleResponseHTTP({
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": '{"score": %s}' % literal},
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                      "total_tokens": 12},
        })
        monkeypatch.setattr(cm.httpx, "Client", http)

        client = DeepSeekClient(api_key="k", base_url="https://x.com", model="m")
        with pytest.raises(LLMError):
            client.chat_json(system="s", user="u")
        assert http.calls == 1

    def test_non_finite_constant_has_explicit_error_message(self, monkeypatch):
        """Minor 4-волны: ValueError от parse_constant → явный LLMError.

        Раньше путь держался на catch-all `except Exception` и давал
        «Неожиданная ошибка…». Теперь — отдельная ветка без retry.
        """
        import newsbot.llm.client as cm

        http = _SingleResponseHTTP({
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": '{"score": NaN}'},
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                      "total_tokens": 12},
        })
        monkeypatch.setattr(cm.httpx, "Client", http)

        client = DeepSeekClient(api_key="k", base_url="https://x.com", model="m")
        with pytest.raises(LLMError) as exc_info:
            client.chat_json(system="s", user="u")

        assert type(exc_info.value) is LLMError  # не LLMTruncationError
        msg = str(exc_info.value)
        assert "Недопустимая JSON-константа в ответе" in msg
        assert "недопустимая JSON-константа: NaN" in msg  # текст из ValueError
        assert "Неожиданная ошибка" not in msg
        assert http.calls == 1  # без retry


# ─────────────────────────────────────────────────────────────────
# MINOR 4: shutdown прекращает отправку (burst при обнулённых паузах)
# ─────────────────────────────────────────────────────────────────


class TestTelegramShutdown:
    @staticmethod
    def _delivery(event):
        from newsbot.delivery.telegram import TelegramDelivery
        return TelegramDelivery(token="tok", chat_id="123", shutdown_event=event)

    def test_send_batch_makes_no_http_calls_on_shutdown(self):
        import threading
        from unittest.mock import patch

        event = threading.Event()
        event.set()
        td = self._delivery(event)
        items = [make_item(url=f"https://e.com/{i}") for i in range(3)]

        with patch.object(td._http, "post") as mock_post:
            delivered = td.send_batch(items)

        assert delivered == ([], [])       # всё остаётся pending
        assert mock_post.call_count == 0

    def test_send_batch_stops_after_current_item(self):
        import threading
        from unittest.mock import patch

        event = threading.Event()
        td = self._delivery(event)
        items = [make_item(url=f"https://e.com/{i}") for i in range(3)]

        def _send_one(item):
            event.set()  # SIGTERM пришёл во время отправки текущего item

        with patch.object(td, "_send_one", side_effect=_send_one) as sent, \
             patch.object(td, "_wait"):
            delivered = td.send_batch(items)

        assert delivered == (["https://e.com/0"], [])  # текущий доставлен
        assert sent.call_count == 1              # остаток — pending

    def test_api_call_raises_on_shutdown_without_http(self):
        import threading
        from unittest.mock import patch

        from newsbot.delivery.telegram import TelegramDeliveryError

        event = threading.Event()
        event.set()
        td = self._delivery(event)

        with patch.object(td._http, "post") as mock_post:
            with pytest.raises(TelegramDeliveryError):
                td._api_call("sendMessage", {"chat_id": "123", "text": "hi"})

        assert mock_post.call_count == 0


# ─────────────────────────────────────────────────────────────────
# Cost-guard: лог-шум (WARNING один раз за день), INFO на повторном
# превышении и guard между чанками LLM-дедупа
# ─────────────────────────────────────────────────────────────────


class _DedupClient:
    """Фейковый клиент дедупа: отдаёт duplicates, копит usage, считает вызовы."""

    def __init__(self, dup_ids=None, usage=None):
        self.calls = 0
        self.usage_stats = UsageStats(model="deepseek-flash")
        self._dup_ids = list(dup_ids or [])
        self._usage = usage

    def chat_json(self, system, user):
        self.calls += 1
        if self._usage:
            self.usage_stats.add_response({"usage": self._usage})
        return {"duplicates": list(self._dup_ids)}


def _warnings(records, needle):
    return [
        r for r in records
        if r.levelno == logging.WARNING and needle in r.getMessage()
    ]


class TestCostLogNoise:
    def test_repeated_cost_block_warns_once_per_day(self, tmp_storage, caplog):
        """WARNING (лимит + pending-напоминание) — только в первом блоке дня."""
        tmp_storage.set_meta(_today_key(), "0.75")
        # pending непустой — иначе pending-напоминание молчит
        tmp_storage.save(make_item(url="https://e.com/p1"))
        tmp_storage.set_state(["https://e.com/p1"], "pending")

        client = _CountingClient(raise_if_called=True)
        flt = _cost_filter(client, tmp_storage, limit=0.50)
        batch = [
            make_item(url="https://e.com/1"),
            make_item(url="https://e.com/2"),
        ]

        with caplog.at_level(logging.DEBUG):
            flt.evaluate_batch(batch)
            first = list(caplog.records)
            caplog.clear()
            flt.evaluate_batch(batch)
            second = list(caplog.records)
            caplog.clear()
            flt.evaluate_batch(batch)
            third = list(caplog.records)

        # Первый блок: ровно по одному WARNING на каждое сообщение
        assert len(_warnings(first, "daily cost limit reached")) == 1
        assert len(_warnings(first, "в pending накоплено")) == 1
        # Повторы — не WARNING
        assert _warnings(second, "daily cost limit") == []
        assert _warnings(second, "в pending накоплено") == []
        assert _warnings(third, "daily cost limit") == []
        assert _warnings(third, "в pending накоплено") == []
        # ... но диагностика не потеряна: pending виден в DEBUG
        assert any(
            r.levelno == logging.DEBUG and "в pending накоплено" in r.getMessage()
            for r in third
        )

    def test_repeated_exceed_logs_info_with_costs(self, tmp_storage, caplog):
        """Повторное превышение при установленном флаге → INFO с cost/limit."""
        tmp_storage.set_meta(_today_key(), "0.75")
        client = _CountingClient(raise_if_called=True)
        flt = _cost_filter(client, tmp_storage, limit=0.50)

        assert flt._check_daily_cost() is False  # первый раз — WARNING + флаг
        caplog.clear()
        with caplog.at_level(logging.INFO):
            assert flt._check_daily_cost() is False

        recs = [r for r in caplog.records if "reached again" in r.getMessage()]
        assert len(recs) == 1
        assert recs[0].levelno == logging.INFO
        msg = recs[0].getMessage()
        assert "0.75" in msg and "0.50" in msg


class TestDedupChunkCostGuard:
    def test_limit_exhausted_mid_cycle_stops_between_chunks(self, tmp_storage, caplog):
        """Лимит исчерпан после 1-го чанка → остальные чанки не идут в LLM."""
        # pending непустой — иначе pending-напоминание молчит
        tmp_storage.save(make_item(url="https://e.com/p1"))
        tmp_storage.set_state(["https://e.com/p1"], "pending")

        client = _DedupClient(
            dup_ids=[2],
            usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
        )
        flt = _cost_filter(client, tmp_storage, limit=0.50)
        chunk = LLMFilter._DEDUP_CHUNK_SIZE
        items = [
            make_item(url=f"https://e.com/d{i}", title=f"T{i}")
            for i in range(chunk * 2)
        ]

        with caplog.at_level(logging.WARNING):
            dups = flt.detect_duplicates(items, [])

        # только 1 чанк ушёл в LLM (2-й отсечён guard'ом внутри цикла)
        assert client.calls == 1
        # находки 1-го чанка сохранены (0-based индекс 1 = 2-й item чанка)
        assert dups == {1}
        # Minor-фикс: чанк-guard тоже напоминает про pending (иначе при лимите,
        # исчерпанном посреди цикла, WARNING за день не появился бы ни разу)
        assert len(_warnings(caplog.records, "в pending накоплено")) == 1


# ─────────────────────────────────────────────────────────────────
# Major 1: cost-guard не должен молча превращаться в no-op
# ─────────────────────────────────────────────────────────────────


class TestCostGuardObservability:
    def test_warns_when_model_prices_missing(self, caplog):
        """Модель не в pricing + нули в клиенте → WARNING при старте."""
        client = _CountingClient()  # usage_stats с нулевыми ценами
        with caplog.at_level(logging.WARNING):
            flt = LLMFilter(
                {
                    "enabled": True,
                    "model": "unknown-model",
                    "api_key": "k",
                    "base_url": "https://x.com",
                    "push_urgent_only": False,
                    "pricing": {"deepseek-flash": {"input": 0.27, "output": 1.10}},
                    "daily_cost_limit": 0.5,
                },
                client=client,
            )

        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("цены модели" in m and "unknown-model" in m for m in msgs), msgs
        # Поведение не изменилось — guard по-прежнему работает (cost = 0.0)
        assert flt._daily_cost_limit == 0.5

    def test_no_warning_when_prices_present(self, caplog):
        client = _CountingClient()
        with caplog.at_level(logging.WARNING):
            LLMFilter(
                {
                    "enabled": True,
                    "model": "deepseek-flash",
                    "api_key": "k",
                    "push_urgent_only": False,
                    "pricing": {"deepseek-flash": {"input": 0.27, "output": 1.10}},
                    "daily_cost_limit": 0.5,
                },
                client=client,
            )
        assert not _warnings(caplog.records, "цены модели")

    def test_no_warning_when_limit_disabled(self, caplog):
        """daily_cost_limit=0 → guard выключен, шума нет."""
        client = _CountingClient()
        with caplog.at_level(logging.WARNING):
            LLMFilter(
                {
                    "enabled": True,
                    "model": "unknown-model",
                    "api_key": "k",
                    "push_urgent_only": False,
                    "daily_cost_limit": 0,
                },
                client=client,
            )
        assert not _warnings(caplog.records, "цены модели")

    def test_broken_pricing_entries_warn_not_crash(self, caplog):
        """Битые записи pricing (скаляр / нечисловые цены) → warning, не continue."""
        client = _CountingClient()
        with caplog.at_level(logging.WARNING):
            flt = LLMFilter(
                {
                    "enabled": True,
                    "model": "m",
                    "api_key": "k",
                    "push_urgent_only": False,
                    "pricing": {
                        "m": "oops",
                        "n": {"input": "x", "output": 1},
                    },
                    "daily_cost_limit": 0,
                },
                client=client,
            )
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("llm.pricing['m']" in m for m in msgs), msgs
        assert any("llm.pricing['n']" in m for m in msgs), msgs
        assert flt._pricing == {}


# ─────────────────────────────────────────────────────────────────
# Major 2: невалидные числовые поля llm.* не отключают фильтр
# ─────────────────────────────────────────────────────────────────


class TestNumericFieldValidation:
    def test_bad_thresholds_fall_back_and_filter_works(self, caplog):
        client = _CountingClient()
        with caplog.at_level(logging.WARNING):
            flt = LLMFilter(
                {
                    "enabled": True,
                    "model": "m",
                    "api_key": "k",
                    "base_url": "https://x.com",
                    "push_urgent_only": False,
                    "push_threshold": "high",
                    "digest_threshold": None,
                },
                client=client,
            )

        assert flt.push_threshold == 7
        assert flt.digest_threshold == 5
        assert _warnings(caplog.records, "push_threshold")
        assert _warnings(caplog.records, "digest_threshold")

        results = flt.evaluate_batch([
            make_item(url="https://e.com/1"),
            make_item(url="https://e.com/2"),
        ])
        assert len(results) == 2
        assert all(r.ok for r in results)
        assert len(client.calls) == 1

    def test_negative_max_retries_clamped_to_zero(self, caplog):
        with caplog.at_level(logging.WARNING):
            flt = LLMFilter({
                "enabled": True,
                "model": "m",
                "api_key": "k",
                "push_urgent_only": False,
                "max_retries": -3,
                "daily_cost_limit": 0,
            })
        try:
            assert flt.client.max_retries == 0
        finally:
            flt.client.close()
        assert _warnings(caplog.records, "max_retries")

    def test_negative_daily_cost_limit_disables_guard(self, caplog):
        client = _CountingClient()
        with caplog.at_level(logging.WARNING):
            flt = LLMFilter(
                {
                    "enabled": True,
                    "model": "m",
                    "api_key": "k",
                    "push_urgent_only": False,
                    "daily_cost_limit": -5,
                },
                client=client,
            )
        assert flt._daily_cost_limit == 0.0
        assert _warnings(caplog.records, "daily_cost_limit")


# ─────────────────────────────────────────────────────────────────
# Non-finite числа в llm.* (Major: int(inf) → OverflowError)
# ─────────────────────────────────────────────────────────────────


class TestNonFiniteNumericFields:
    def test_max_tokens_inf_falls_back_to_default(self, caplog):
        """Major-регресс: max_tokens: .inf → дефолт, а не OverflowError при старте."""
        with caplog.at_level(logging.WARNING):
            flt = LLMFilter({
                "enabled": True,
                "model": "m",
                "api_key": "k",
                "push_urgent_only": False,
                "max_tokens": float("inf"),
                "daily_cost_limit": 0,
            })
        try:
            assert flt.client.max_tokens == 4096
        finally:
            flt.client.close()
        assert _warnings(caplog.records, "max_tokens")

    def test_push_threshold_inf_falls_back_to_default(self, caplog):
        """push_threshold: .inf → дефолт 7 + warning, фильтр работает."""
        client = _CountingClient()
        with caplog.at_level(logging.WARNING):
            flt = LLMFilter(
                {
                    "enabled": True,
                    "model": "m",
                    "api_key": "k",
                    "push_urgent_only": False,
                    "push_threshold": float("inf"),
                },
                client=client,
            )
        assert flt.push_threshold == 7
        assert _warnings(caplog.records, "push_threshold")

    def test_daily_cost_limit_nan_disables_guard(self, caplog):
        """Minor-регресс: daily_cost_limit: .nan → guard выключен (дефолт 0).

        Иначе `nan <= 0` == False → guard «активен», а `current < nan` всегда
        False → ВСЕ LLM-вызовы блокируются каждый цикл.
        """
        client = _CountingClient()
        with caplog.at_level(logging.WARNING):
            flt = LLMFilter(
                {
                    "enabled": True,
                    "model": "m",
                    "api_key": "k",
                    "push_urgent_only": False,
                    "daily_cost_limit": float("nan"),
                },
                client=client,
            )
        assert flt._daily_cost_limit == 0.0
        assert _warnings(caplog.records, "daily_cost_limit")
        # guard действительно выключен — вызов не блокируется
        assert flt._check_daily_cost() is True

    def test_nan_pricing_entry_skipped(self, caplog, tmp_storage):
        """Minor-регресс: pricing.<model>.input = .nan → запись пропущена.

        Иначе cost=nan → персистентный meta "nan" → cost-guard блокирует все
        вызовы до смены дня.
        """
        client = _CountingClient()
        with caplog.at_level(logging.WARNING):
            flt = LLMFilter(
                {
                    "enabled": True,
                    "model": "m",
                    "api_key": "k",
                    "push_urgent_only": False,
                    "daily_cost_limit": 1.0,
                    "pricing": {
                        "m": {"input": float("nan"), "output": 1.0},
                        "ok": {"input": 1.0, "output": 2.0},
                    },
                },
                client=client,
                storage=tmp_storage,
            )
        assert "m" not in flt._pricing
        assert flt._pricing["ok"] == {"input": 1.0, "output": 2.0}
        assert _warnings(caplog.records, "pricing")

        # NaN не должен просочиться в дневной счётчик meta
        flt._accumulate_daily_cost(1000, 1000)
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        stored = float(tmp_storage.get_meta(f"llm_daily_cost_{today}") or "0")
        assert stored == stored  # not NaN
        assert stored == 0.0  # цена модели неизвестна → в клиенте нули


# ─────────────────────────────────────────────────────────────────
# Minor 3: LLMTruncationError вместо строкового детекта
# ─────────────────────────────────────────────────────────────────


class _TypedTruncationClient:
    """Бросает LLMTruncationError с текстом БЕЗ маркеров length/truncat."""

    def __init__(self):
        self.user_sizes = []
        self.usage_stats = UsageStats(model="m")

    def chat_json(self, system, user):
        n = user.count("<|data:")
        self.user_sizes.append(n)
        if len(self.user_sizes) == 1:
            raise LLMTruncationError("ответ обрезан по токенам")
        return {
            "results": [
                {"id": i, "score": 6, "urgent": False, "summary": "S", "reason": "r"}
                for i in range(1, n + 1)
            ]
        }


class TestTruncationErrorType:
    def test_split_triggered_by_exception_type(self):
        """Minor 3: split срабатывает по типу исключения, а не по тексту."""
        client = _TypedTruncationClient()
        flt = LLMFilter(
            {"enabled": True, "model": "m", "api_key": "k",
             "base_url": "https://x.com", "push_urgent_only": False},
            client=client,
        )
        items = [make_item(url=f"https://e.com/{i}") for i in range(4)]
        results = flt.evaluate_batch(items)

        assert len(results) == 4
        assert all(r.ok for r in results)
        assert client.user_sizes == [4, 2, 2]

    def test_client_raises_llmtruncationerror(self, monkeypatch):
        """client.py: finish_reason=length → LLMTruncationError (подкласс LLMError)."""
        import newsbot.llm.client as cm

        class _FakeResponse:
            status_code = 200

            def json(self):
                return {
                    "choices": [{
                        "finish_reason": "length",
                        "message": {"content": '{"score": 8'},
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20,
                              "total_tokens": 30},
                }

            def raise_for_status(self):
                pass

        class _LengthHTTP:
            is_closed = False

            def __init__(s, **kw):
                pass

            def post(s, url, **kwargs):
                return _FakeResponse()

            def close(s):
                pass

        orig = cm.httpx.Client
        cm.httpx.Client = _LengthHTTP  # type: ignore
        try:
            client = DeepSeekClient(api_key="k", base_url="https://x.com", model="m")
            with pytest.raises(LLMTruncationError) as exc_info:
                client.chat_json(system="s", user="u")
            assert isinstance(exc_info.value, LLMError)
            assert "finish_reason=length" in str(exc_info.value)
        finally:
            cm.httpx.Client = orig


# ─────────────────────────────────────────────────────────────────
# Minor 4: float-индексы в дедуп-ответе
# ─────────────────────────────────────────────────────────────────


class TestDedupFloatIndices:
    def _filter(self):
        return LLMFilter(
            {"enabled": True, "model": "m", "api_key": "k",
             "push_urgent_only": False},
            client=_CountingClient(),
        )

    def test_float_indices_recognized(self):
        """[1.0, 2.0] → дубли найдены (1.0 = лидер чанка, 2.0 → индекс 1)."""
        flt = self._filter()
        dups = flt._parse_dedup_response(
            {"duplicates": [1.0, 2.0]}, chunk_len=3, chunk_offset=0, recent=[],
        )
        assert dups == {1}

    def test_non_integral_and_nan_floats_ignored(self):
        flt = self._filter()
        dups = flt._parse_dedup_response(
            {"duplicates": [2.5, float("nan"), float("inf"), 3.0]},
            chunk_len=3,
            chunk_offset=0,
            recent=[],
        )
        assert dups == {2}

    def test_float_out_of_range_ignored(self):
        flt = self._filter()
        assert flt._parse_dedup_response(
            {"duplicates": [4.0]}, chunk_len=3, chunk_offset=0, recent=[],
        ) == set()


