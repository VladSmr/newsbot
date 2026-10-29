"""
Тесты push-cap: лимит push-уведомлений за цикл.

Проверяется:
  (a) 5 push-eligible, cap=3 → 3 push + 2 deferred
  (b) cap=0 → безлимит (back-compat)
  (c) цикл-левел: два батча 2+2, cap=3 → 3 push + 1 deferred
  (d) порядок отбора детерминирован (score desc, fetched_at asc)
  (e) валидатор границ config: max_pushes_per_cycle в диапазоне [0..50]
  (f) ok=False (LLM-сбой) → _process_batch НЕ падает, items обрабатываются штатно
"""
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

# For test (f): patch setup_logging to avoid config.py writing to stderr
try:
    import newsbot.config as config_mod
    config_mod.setup_logging = lambda *a, **kw: None  # type: ignore[assignment]
except Exception:
    pass

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from newsbot.collector import NewsItem
from newsbot.llm.filter import FilterResult
from newsbot.storage import Storage
from newsbot.pipeline import NewsPipeline


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def make_item(url, title="Test", score=8, urgent=True, fetched_at=None,
               has_llm_score=True):
    """has_llm_score=False → fresh item (llm_score=None, пойдёт через evaluate_batch)."""
    item = NewsItem(
        source="TestSource",
        url=url,
        title=title,
        summary="summary",
        published="2024-01-01",
        published_parsed=None,
    )
    item.llm_score = score if has_llm_score else None
    item.llm_summary = "summary"
    item.llm_reason = f"test score={score}"
    item.llm_urgent = urgent
    # Fix 3: fetched_at может быть datetime или str — консистентность
    if fetched_at is None:
        item.fetched_at = "2024-01-01T10:00:00"
    elif hasattr(fetched_at, "isoformat"):  # datetime
        item.fetched_at = fetched_at
    else:
        item.fetched_at = fetched_at
    return item


class FakeDelivery:
    """Мок доставки: запоминает отправленные items, не шлёт в сеть."""
    def __init__(self):
        self.sent = []

    def send_batch(self, items):
        self.sent.extend([i.url for i in items])
        return (self.sent[:], [])

    def send_text(self, text):
        return True

    def close(self):
        pass


class FakeLLMFilter:
    """Fake-фильтр: создаёт FilterResult из score. ok=False эмулирует LLM-сбой."""
    def __init__(self, urgent_only=False,
                 # Если error_urls задан — только эти url получают ok=False.
                 # Если error_urls=None и force_ok_false=True — ВСЕ fresh-items ok=False.
                 error_urls=None, force_ok_false=False):
        self.push_threshold = 7
        self.digest_threshold = 5
        self.push_urgent_only = urgent_only
        self._error_urls = set(error_urls) if error_urls else set()
        self._force_ok_false = force_ok_false

    def evaluate_batch(self, items, feedback_context=None):
        results = []
        for item in items:
            is_fresh = item.llm_score is None  # no pre-existing score → fresh
            if (is_fresh and (item.url in self._error_urls or self._force_ok_false)):
                results.append(FilterResult(
                    ok=False,
                    cost_blocked=True,
                    score=None,
                    summary="",
                    reason="test-cost-blocked",
                    urgent=False,
                ))
            elif is_fresh:
                results.append(FilterResult(
                    ok=True,
                    score=item.llm_score or 8,
                    summary=getattr(item, "llm_summary", "") or "s",
                    reason=getattr(item, "llm_reason", "") or "",
                    urgent=bool(getattr(item, "llm_urgent", True)),
                ))
            else:
                # retry item (llm_score set): synthesize ok result
                results.append(FilterResult(
                    ok=True,
                    score=item.llm_score,
                    summary=getattr(item, "llm_summary", "") or "s",
                    reason=getattr(item, "llm_reason", "") or "",
                    urgent=bool(getattr(item, "llm_urgent", True)),
                ))
        return results


def _cleanup_db(path):
    import time
    for attempt in range(3):
        try:
            Path(path).unlink(missing_ok=True)
            Path(path + "-wal").unlink(missing_ok=True)
            Path(path + "-shm").unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt < 2:
                time.sleep(0.1)


def fresh_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    Path(path).unlink()
    return path, lambda: _cleanup_db(path)


def make_pipeline(db_path, llm_filter=None, extra_config=None):
    """Создаёт NewsPipeline с fake-компонентами и optionally push-cap."""
    cfg = {
        "llm": {
            "push_threshold": 7,
            "digest_threshold": 5,
            "push_urgent_only": False,
            "max_per_cycle": 100,
            "max_pushes_per_cycle": 0,  # override below
        },
        "telegram": {"token": "test-token", "chat_id": "0"},
        "digest": {"enabled": False},
        "dedup": {"cross_source": False},
    }
    if extra_config:
        for section, values in extra_config.items():
            cfg.setdefault(section, {}).update(values)
    # override llm section with any llm-specific config
    if extra_config and "llm" in extra_config:
        for k, v in extra_config["llm"].items():
            cfg["llm"][k] = v

    p = NewsPipeline.__new__(NewsPipeline)
    p.config = cfg
    p.storage = Storage(db_path)
    p.delivery = FakeDelivery()
    p.llm_filter = llm_filter
    p._llm_calls_left = 100
    p._max_per_cycle = cfg["llm"]["max_per_cycle"]
    p._max_retries = 3
    p.sources = []
    p._feedback_context = None  # needed by evaluate_batch feedback_context param
    return p


# ─────────────────────────────────────────────────────────────────
# Тест (a): 5 push-eligible, cap=3 → 3 push + 2 deferred
# ─────────────────────────────────────────────────────────────────

def test_push_cap_basic():
    """
    5 push-eligible items, max_pushes_per_cycle=3.
    Ожидание: 3 pushed, 2 deferred (url2, url3).
    """
    db_path, cleanup = fresh_db()
    try:
        pipeline = make_pipeline(
            db_path,
            llm_filter=FakeLLMFilter(),
            extra_config={"llm": {"max_pushes_per_cycle": 3}},
        )
        # items: all push-eligible (score=9, urgent=True)
        items = [
            make_item(f"u{i}", score=9, urgent=True) for i in range(5)
        ]
        # переводим в pending
        pipeline.storage.set_state([i.url for i in items], "pending")

        # вызываем _process_batch напрямую
        push_ref = [0]
        pipeline._process_batch(
            items, budget_override=len(items),
            push_count_ref=push_ref,
        )

        assert push_ref[0] == 3, f"Expected 3 pushes, got {push_ref[0]}"
        # deferred: те что НЕ в sent (overflow по cap)
        deferred = [u for u in [f"u{i}" for i in range(5)] if u not in pipeline.delivery.sent]
        assert len(deferred) == 2, f"Expected 2 deferred, got {deferred}"
        assert len(pipeline.delivery.sent) == 3, f"Expected 3 sent, got {pipeline.delivery.sent}"
    finally:
        cleanup()


# ─────────────────────────────────────────────────────────────────
# Тест (b): cap=0 → безлимит (back-compat)
# ─────────────────────────────────────────────────────────────────

def test_push_cap_zero_means_unlimited():
    """
    max_pushes_per_cycle=0: все 5 push-eligible доставляются, deferred=0.
    """
    db_path, cleanup = fresh_db()
    try:
        pipeline = make_pipeline(
            db_path,
            llm_filter=FakeLLMFilter(),
            extra_config={"llm": {"max_pushes_per_cycle": 0}},
        )
        items = [
            make_item(f"u{i}", score=9, urgent=True) for i in range(5)
        ]
        pipeline.storage.set_state([i.url for i in items], "pending")

        push_ref = [0]
        pipeline._process_batch(
            items, budget_override=len(items),
            push_count_ref=push_ref,
        )

        assert push_ref[0] == 5, f"Expected 5 pushes (unlimited), got {push_ref[0]}"
        assert len(pipeline.delivery.sent) == 5
    finally:
        cleanup()


# ─────────────────────────────────────────────────────────────────
# Тест (c): цикл-левел — два батча 2+2 при cap=3 → 3 push + 1 deferred
# ─────────────────────────────────────────────────────────────────

def test_push_cap_is_cycle_level():
    """
    Имитация двух вызовов _process_batch в одном цикле.
    Батч1: 2 push-eligible, Батч2: 2 push-eligible, cap=3.
    Ожидание: 3 push (2+1), 1 deferred (overflow from batch2).
    """
    db_path, cleanup = fresh_db()
    try:
        pipeline = make_pipeline(
            db_path,
            llm_filter=FakeLLMFilter(),
            extra_config={"llm": {"max_pushes_per_cycle": 3}},
        )
        batch1 = [make_item(f"b1_u{i}", score=9, urgent=True) for i in range(2)]
        batch2 = [make_item(f"b2_u{i}", score=9, urgent=True) for i in range(2)]
        all_items = batch1 + batch2
        pipeline.storage.set_state([i.url for i in all_items], "pending")

        # ЕДИНЫЙ счётчик на цикл
        push_ref = [0]

        pipeline._process_batch(batch1, budget_override=2, push_count_ref=push_ref)
        assert push_ref[0] == 2, f"After batch1: expected 2, got {push_ref[0]}"

        pipeline._process_batch(batch2, budget_override=2, push_count_ref=push_ref)
        assert push_ref[0] == 3, f"After batch2: expected 3, got {push_ref[0]}"

        # 3 отправлено (2+1), 1 deferred (overflow from batch2)
        assert len(pipeline.delivery.sent) == 3, f"Expected 3 sent, got {pipeline.delivery.sent}"
        b2_candidates = [f"b2_u{i}" for i in range(2)]
        b2_overflow = [u for u in b2_candidates if u not in pipeline.delivery.sent]
        assert len(b2_overflow) == 1, f"Expected 1 deferred from batch2, got {b2_overflow}"
    finally:
        cleanup()


# ─────────────────────────────────────────────────────────────────
# Тест (d): порядок отбора детерминирован (score desc, fetched_at asc)
# ─────────────────────────────────────────────────────────────────

def test_push_cap_deterministic_order():
    """
    При равном score — старые (меньший fetched_at) раньше.
    Items: all score=9, fetched_at increasing.
    Cap=2 → старшие 2 уходят в push, остальные deferred.
    """
    db_path, cleanup = fresh_db()
    try:
        pipeline = make_pipeline(
            db_path,
            llm_filter=FakeLLMFilter(),
            extra_config={"llm": {"max_pushes_per_cycle": 2}},
        )
        # fetched_at: чем меньше строка, тем старше
        items = [
            make_item(f"u{i}", score=9, urgent=True, fetched_at=f"2024-01-01T{10+i:02d}:00:00")
            for i in range(4)
        ]
        pipeline.storage.set_state([i.url for i in items], "pending")

        push_ref = [0]
        pipeline._process_batch(
            items, budget_override=len(items),
            push_count_ref=push_ref,
        )

        # u0 (10:00) и u1 (11:00) — старшие → в push
        assert len(pipeline.delivery.sent) == 2
        assert "u0" in pipeline.delivery.sent, f"Oldest should be in push: {pipeline.delivery.sent}"
        assert "u1" in pipeline.delivery.sent
        assert "u2" not in pipeline.delivery.sent, "Newer should be deferred"
        assert "u3" not in pipeline.delivery.sent, "Newest should be deferred"
    finally:
        cleanup()


# ─────────────────────────────────────────────────────────────────
# Тест (e): валидатор границ max_pushes_per_cycle
# ─────────────────────────────────────────────────────────────────

def test_config_validator_bounds():
    """max_pushes_per_cycle: валидные [0..50] проходят, вне диапазона → дефолт 0.

    Использует настоящий load_config с tmp yaml — реальная валидация config.py.
    """
    import yaml
    import newsbot.config as config_mod

    # Базовый yaml с минимальным набором, чтобы load_config не упал
    _BASE_YAML = {
        "sources": [],
        "database": {"path": ":memory:"},
        "llm": {
            "push_threshold": 7,
            "digest_threshold": 5,
            "push_urgent_only": False,
            "max_per_cycle": 100,
            "api_key": "test-key",
            "model": "deepseek-flash",
        },
        "telegram": {"token": "test-token"},
    }

    def _load_with_cap(cap_value):
        """Грузит конфиг через настоящий load_config с переопределённым max_pushes_per_cycle."""
        cfg = dict(_BASE_YAML)
        cfg["llm"] = dict(_BASE_YAML["llm"])
        cfg["llm"]["max_pushes_per_cycle"] = cap_value
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                yaml.dump(cfg, fh)
            # suppress logging for cleaner test output
            import logging
            logging.getLogger("newsbot.config").setLevel(logging.CRITICAL + 1)
            try:
                loaded = config_mod.load_config(path)
            finally:
                logging.getLogger("newsbot.config").setLevel(logging.WARNING)
            return loaded.get("llm", {}).get("max_pushes_per_cycle", 0)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    # Валидные значения — проходят как есть
    assert _load_with_cap(0) == 0, "cap=0 валиден"
    assert _load_with_cap(3) == 3, "cap=3 валиден"
    assert _load_with_cap(50) == 50, "cap=50 валиден (верхняя граница)"

    # Вне диапазона → дефолт 0
    assert _load_with_cap(-1) == 0, "cap=-1 → дефолт 0"
    assert _load_with_cap(51) == 0, "cap=51 → дефолт 0 (выше 50)"
    assert _load_with_cap(999) == 0, "cap=999 → дефолт 0"

    # Мусор → дефолт 0
    assert _load_with_cap("abc") == 0, "cap='abc' → дефолт 0"
    assert _load_with_cap(None) == 0, "cap=None → дефолт 0"
    assert _load_with_cap([]) == 0, "cap=[] → дефолт 0"


def test_ok_false_no_crash():
    """FilterResult(ok=False) при push-cap: _process_batch НЕ падает.

    Critical: без ok-guard перед route() → route(None, ...) → TypeError.
    Механика: has_llm_score=False → fresh-путь → evaluate_batch → ok=False.
    guard `if not res.ok: continue` срабатывает → route() НЕ вызывается.
    Counter не трогается (ok=False не считается как push).
    """
    db_path, cleanup = fresh_db()
    try:
        pipeline = make_pipeline(
            db_path,
            # force_ok_false=True: ВСЕ fresh-items получают ok=False от evaluate_batch
            llm_filter=FakeLLMFilter(force_ok_false=True),
            extra_config={"llm": {"max_pushes_per_cycle": 3}},
        )
        # has_llm_score=False: fresh item (пойдёт через evaluate_batch).
        # force_ok_false=True: FakeLLMFilter вернёт ok=False для всех fresh-items.
        items = [
            make_item(f"u{i}", score=9, urgent=True, has_llm_score=False)
            for i in range(3)
        ]
        pipeline.storage.set_state([i.url for i in items], "pending")

        push_ref = [0]
        # Critical: route(None, ...) без guard → TypeError.
        # С guard: ok=False → _apply_result без route(), counter=0.
        pipeline._process_batch(
            items, budget_override=len(items),
            push_count_ref=push_ref,
        )
        # ok=False не должен инкрементировать счётчик
        assert push_ref[0] == 0, (
            f"ok=False не должны считаться как push, counter={push_ref[0]}"
        )
        # _process_batch вернул управление (не упал с TypeError)
    finally:
        cleanup()