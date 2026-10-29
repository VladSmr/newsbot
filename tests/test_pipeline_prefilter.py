"""
Тесты поведения префильтра (Задача 2 / V2) и empty-summary оркестрации (Задача 3 / V3).

Запуск: python -m pytest tests/test_pipeline_prefilter.py -v
"""
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from newsbot.collector import NewsItem
from newsbot.llm.filter import FilterResult
from newsbot.pipeline import NewsPipeline, _prefilter_item
from newsbot.storage import Storage


# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────

class FakeLLMFilter:
    """Fake-фильтр: все items проходят с score=8 (push)."""
    def __init__(self):
        self.push_threshold = 7
        self.digest_threshold = 5
        self.push_urgent_only = False
        self._dedup_seen: list = []

    def evaluate_batch(self, items, feedback_context=None):
        self._dedup_seen.extend([i.url for i in items])
        return [
            FilterResult(ok=True, score=8, summary=f"Sum: {i.title[:20]}",
                         reason="test", urgent=False, location="Moscow, Russia",
                         title_en="Test Title")
            for i in items
        ]

    def detect_duplicates(self, candidates, recent):
        """Fake: все кандидаты уникальны — возвращает пустой set."""
        self._dedup_seen.extend([c["url"] for c in candidates])
        return set()


class FakeLLMFilterEmptySummary:
    """Fake-фильтр: возвращает score=2, пустой summary (имитация phase2-budget-cut)."""
    def __init__(self):
        self.push_threshold = 7
        self.digest_threshold = 5
        self.push_urgent_only = False

    def evaluate_batch(self, items, feedback_context=None):
        return [
            FilterResult(ok=True, score=2, summary="", reason="low score",
                         urgent=False, location="", title_en=None)
            for i in items
        ]

    def get_balance(self):
        class _Bal:
            is_available = True
            total_balance = 100.0
            used_today = 0.0
            remaining = 100.0
            currency = "USD"
        return _Bal()


def _cleanup_db(path):
    """Windows-safe cleanup: закрывает соединение и удаляет DB + WAL."""
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
    """Возвращает путь к временной БД и cleanup-функцию."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    Path(path).unlink()  # SQLite сам создаст
    return path, lambda: _cleanup_db(path)


def _insert_item(db_path, url, title, summary, source="test", state="new",
                 llm_score=None, llm_summary=None, llm_reason=None,
                 published=None, fetched_at=None):
    """Вставляет item напрямую в БД (минуя collector)."""
    if published is None:
        published = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    if fetched_at is None:
        fetched_at = datetime.now(timezone.utc).isoformat()
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO items "
        "(url,title,summary,source,state,state_at,fetched_at,retry_count,"
        "llm_score,llm_summary,llm_reason,published) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (url, title, summary, source, state, now_iso, fetched_at, 0,
         llm_score, llm_summary, llm_reason, published),
    )
    conn.commit()
    conn.close()


def _mk_pipeline(db_path, llm_filter=None, extra_config=None):
    config = {
        "sources": [],
        "database": {"path": db_path},
        "telegram": {"token": "", "chat_id": ""},
        "poll_interval_minutes": 15,
        "digest": {
            "enabled": False,
        },
    }
    if extra_config:
        config.update(extra_config)
    p = NewsPipeline(config)
    if llm_filter is not None:
        p.llm_filter = llm_filter
    return p


def _make_item(url, title, summary, source="test", published="2024-01-01"):
    """Creates a NewsItem with required positional fields in dataclass order."""
    return NewsItem(
        source=source,
        url=url,
        title=title,
        summary=summary,
        published=published,
    )


# ─────────────────────────────────────────────────────────────────
# Тесты V2: _prefilter_item (unit)
# ─────────────────────────────────────────────────────────────────

class TestPrefilterUnit:
    def _item(self, title, summary=""):
        import hashlib
        url = "https://ex.com/" + hashlib.md5(title.encode()).hexdigest()[:8]
        return _make_item(url=url, title=title, summary=summary)

    def test_garbage_opinion_no_facts(self):
        """Мнение без цифр/имён/мест — отсеивается."""
        it = self._item(
            "Expert Opinion: economic outlook looks grim",
            summary="Many analysts believe the situation will worsen",
        )
        assert _prefilter_item(it) is True, "Opinion without facts should be filtered"

    def test_garbage_sources_say(self):
        """'Sources say' без фактов — отсеивается."""
        it = self._item(
            "Sources say major changes are coming soon",
            summary="According to sources, something might happen",
        )
        assert _prefilter_item(it) is True, "'Sources say' without facts should be filtered"

    def test_garbage_celebrity_gossip(self):
        """Светская хроника — отсеивается."""
        it = self._item(
            "Celebrity wedding: royal family celebrates in private ceremony",
            summary="Celebrity gossip and royal family updates",
        )
        assert _prefilter_item(it) is True, "Celebrity gossip should be filtered"

    def test_garbage_daily_fuel_price(self):
        """Суточные цены на топливо — отсеиваются (all-lowercase, без proper nouns)."""
        it = self._item(
            "fuel price update: daily fuel price report",
            summary="fuel prices fell across markets yesterday",
        )
        assert _prefilter_item(it) is True, "Daily fuel prices should be filtered"

    def test_garbage_ai_model_release(self):
        """Релиз AI-модели — отсеивается (all-lowercase)."""
        it = self._item(
            "llm model release announced by openai",
            summary="new openai model is now available for developers",
        )
        assert _prefilter_item(it) is True, "AI model release should be filtered"

    def test_real_news_with_numbers_and_location(self):
        """Реальная новость с цифрами/локацией — НЕ отсеивается (идёт в LLM)."""
        it = self._item(
            "Moscow Drone Attack: 5 Shahed drones shot down over Krasnodar region",
            summary="Air defense repelled an attack on the region",
        )
        assert _prefilter_item(it) is False, "Real news with numbers should pass to LLM"

    def test_real_news_just_numbers(self):
        """Новость с числами (без имени/места) — не отсеивается."""
        it = self._item(
            "UAVs attacked a facility — 12 drones shot down, 3 injured",
            summary="Casualties reported",
        )
        assert _prefilter_item(it) is False, "News with numbers should pass"

    def test_urgent_plus_opinion_not_filtered(self):
        """Urgent-маркер + маркер 'мнения' — НЕ отсеивается."""
        it = self._item(
            "Attack on convoy: 10 killed, sources say it was deliberate",
            summary="Emergency evacuation ordered for civilians",
        )
        assert _prefilter_item(it) is False, "Urgent news should NOT be filtered even with opinion markers"

    def test_urgent_catastrophe_not_filtered(self):
        """Катастрофа (casualties, fire, flood) — не отсеивается."""
        it = self._item(
            "Flooding in the area: thousands evacuated, 5 dead",
            summary="State of emergency declared",
        )
        assert _prefilter_item(it) is False, "Catastrophe with casualties should NOT be filtered"

    def test_pure_russian_opinion_forecast(self):
        """Мнение/прогноз на русском без фактов — отсеивается."""
        it = self._item(
            "Экспертное мнение: ситуация ухудшится в ближайшие месяцы",
            summary="Многие эксперты прогнозируют ухудшение",
        )
        assert _prefilter_item(it) is True, "Russian opinion/forecast without facts should be filtered"

    def test_empty_title_summary(self):
        """Пустой title+summary без маркеров — пропускается в LLM."""
        it = _make_item(url="https://ex.com/empty", title="", summary="")
        assert _prefilter_item(it) is False, "Empty item should pass to LLM (no hard marker)"

    def test_real_news_with_date(self):
        """Новость с датой (но без числа/имени) — не отсеивается."""
        it = self._item(
            "Meeting scheduled for January 15 to discuss new policy",
            summary="Government officials will gather on the date",
        )
        assert _prefilter_item(it) is False, "News with date should pass"

    # ── Major-2 антиподы: защита от ложного отсева реальных новостей ──────

    def test_antipode_catastrophe_with_location(self):
        """'ChatGPT outage hits hospital' — НЕ отсеивается (urgent + location)."""
        it = self._item(
            "ChatGPT outage hits hospital systems across three cities",
            summary="Major AI service down, affecting medical facilities",
        )
        assert _prefilter_item(it) is False, "ChatGPT outage with location should NOT be filtered"

    def test_antipode_sources_say_with_proper_noun(self):
        """'Sources say Sudan army advances' — НЕ отсеивается (PN Sudan после 'say')."""
        it = self._item(
            "Sources say Sudan army advances on capital amid escalating tensions",
            summary="Military forces move toward key city as fighting intensifies",
        )
        assert _prefilter_item(it) is False, (
            "Sources say with proper noun (Sudan) should NOT be filtered"
        )

    def test_antipode_russian_sources_with_location(self):
        """'Источники сообщают о ситуации в регионе' — НЕ отсеивается (location + region)."""
        it = self._item(
            "Источники сообщают о ситуации в регионе",
            summary="Местные власти готовят экстренное совещание",
        )
        assert _prefilter_item(it) is False, "Russian 'источники сообщают' with location should NOT be filtered"

    def test_antipode_russian_inflection_forms(self):
        """Падежные формы маркеров локации: область/городе/деревни/крае/селе.
        После ЗАДАЧИ-1: [а-я]* на конце хвоста позволяет матчить все формы."""
        cases = [
            ("Ситуация в области обостряется", "Локальные власти, источники сообщают"),
            ("Источники сообщают о ситуации в городе", "Прогнозы экспертов сообщают"),
            ("Источники сообщают: данные по деревни и области", "Сводка от источников сообщает"),
            ("События в крае развиваются", "Источники сообщают"),
            ("Из села поступают данные", "Источники сообщают"),
        ]
        for title, summary in cases:
            it = self._item(title=title, summary=summary)
            # Все падежные формы: hard_hit=weak, но has_location=True → не отсеивается
            assert _prefilter_item(it) is False, (
                f"Inflection '{title}' should NOT be filtered (has_location)"
            )

    def test_antipode_russian_explosion(self):
        """'Источники сообщают о взрыве' — НЕ отсеивается (urgent-маркер 'взрыв')."""
        it = self._item(
            "Источники сообщают о взрыве на заводе в Воронеже",
            summary="На месте работают экстренные службы",
        )
        assert _prefilter_item(it) is False, "Russian explosion should NOT be filtered"

    def test_antipode_deficit_no_numbers(self):
        """'Дефицит бензина: очереди на АЗС в регионах' — НЕ отсеивается (urgent + location + PN)."""
        it = self._item(
            "Дефицит бензина: очереди на АЗС в регионах растут",
            summary="Водители часами ждут заправку, местные власти не комментируют ситуацию",
        )
        assert _prefilter_item(it) is False, "Deficit with АЗС/регион should NOT be filtered"

    def test_antipode_chatgpt_release_with_proper_noun(self):
        """ChatGPT с датой/цифрой — не отсеивается."""
        it = self._item(
            "ChatGPT-5 launch: OpenAI releases new model with breakthrough capabilities",
            summary="OpenAI announces GPT-5 with enhanced reasoning",
        )
        assert _prefilter_item(it) is False, "ChatGPT release with proper noun should NOT be filtered"

    def test_antipode_expert_analysis_with_facts(self):
        """'Expert analysis confirms poisoning' — НЕ отсеивается (есть proper noun 'Poisoning' не входит в hard, но контент с датой/местом)."""
        it = self._item(
            "Expert analysis confirms poisoning of river near Krasnodar",
            summary="Toxicology analysis reveals chemical contamination",
        )
        assert _prefilter_item(it) is False, "Expert analysis with location should NOT be filtered"

    def test_antipode_technical_analysis_with_numbers(self):
        """Technical analysis с цифрами — не отсеивается."""
        it = self._item(
            "Technical analysis: 15% drop in oil prices expected by March",
            summary="Market analysts predict significant decline",
        )
        assert _prefilter_item(it) is False, "Technical analysis with numbers should NOT be filtered"

    def test_antipode_according_to_sources_with_proper_noun(self):
        """'Exclusive: Gazprom nears deal, according to sources' — НЕ отсеивается (PN Gazprom)."""
        it = self._item(
            "Exclusive: Gazprom nears deal, according to sources close to negotiations",
            summary="Major energy agreement expected soon",
        )
        assert _prefilter_item(it) is False, (
            "Gazprom as proper noun should NOT be filtered even with 'according to sources'"
        )

    def test_antipode_according_to_sources_with_numbers(self):
        """'According to sources' с цифрами — не отсеивается."""
        it = self._item(
            "Merger talks advance, according to sources — 50% premium offered",
            summary="Deal valued at $3 billion",
        )
        assert _prefilter_item(it) is False, "'According to sources' with numbers should NOT be filtered"

    def test_antipode_deficit_with_numbers(self):
        """'Дефицит бензина: цены рекордно выросли' — НЕ отсеивается (hard-маркер + цифры)."""
        it = self._item(
            "Дефицит бензина: цены рекордно выросли на 40% за неделю",
            summary="Автомобилисты жалуются на нехватку топлива",
        )
        assert _prefilter_item(it) is False, "Бензин/дефицит with numbers should NOT be filtered"

    def test_antipode_source_inform_soobshayut_with_numbers(self):
        """'Источники сообщают' с цифрами — не отсеивается."""
        it = self._item(
            "Источники сообщают о рекордном росте цен на нефть — +30% за месяц",
            summary="Энергетический рынок переживает волатильность",
        )
        assert _prefilter_item(it) is False, "'Источники сообщают' with numbers should NOT be filtered"

    # ── Major-1: proper noun detection (до .lower()) ──────────────────────

    def test_proper_noun_russian_uppercase(self):
        """Проверяет путь PN: латинский Gazprom в скобках активирует has_latin=True → has_proper_noun (pipeline.py:583-586)."""
        it = self._item(
            "Источники сообщают: Газпром (Gazprom) представил отчёт",
            summary="Президент компании заявил о намерениях",
        )
        # "Gazprom" — латинский вложенный в PN: has_latin=True сразует has_proper_noun=True в строке 583-586
        assert _prefilter_item(it) is False, "PN from latin embedded in Russian name should bypass prefilter"

    def test_proper_noun_english_uppercase(self):
        """PN с заглавной буквой (Sudan) — обнаруживается, item не отсеивается."""
        it = self._item(
            "Sudan deal announced, market reaction cautious",
            summary="Many analysts predict decline in emerging markets",
        )
        # "Sudan" с upper-after-first (Su) — true PN → НЕ отсеивается
        assert _prefilter_item(it) is False, "English proper noun (Sudan) should bypass prefilter"


# ─────────────────────────────────────────────────────────────────
# Тесты V2: интеграция в pipeline (_process_batch)
# ─────────────────────────────────────────────────────────────────

class TestPrefilterIntegration:
    def _item(self, title, summary="", url=None):
        if url is None:
            import hashlib
            url = "https://ex.com/" + hashlib.md5(title.encode()).hexdigest()[:8]
        return _make_item(url=url, title=title, summary=summary)

    def test_garbage_skipped_no_llm_call(self, tmp_path):
        """Мусор отсеивается ДО LLM (проверяем call_count мока)."""
        db_path = str(tmp_path / "test.db")
        Path(db_path).unlink(missing_ok=True)
        Storage(db_path).close()

        fake_filter = FakeLLMFilter()
        call_count = [0]
        original_eval = fake_filter.evaluate_batch

        def counting_eval(items, feedback_context=None, **kwargs):
            call_count[0] += 1
            return original_eval(items, feedback_context)

        fake_filter.evaluate_batch = counting_eval

        # Item: мнение без фактов → должен отсеяться
        bad_item = self._item(
            "Expert Opinion: Russia will face economic collapse by 2030",
            summary="Analysts predict major downturn soon",
        )
        _insert_item(db_path, bad_item.url, bad_item.title, bad_item.summary)

        # Item: реальная новость с фактами → идёт в LLM
        good_item = self._item(
            "Moscow Drone Attack: 5 Shahed drones shot down over Krasnodar",
            summary="Air defense repelled an attack",
        )
        _insert_item(db_path, good_item.url, good_item.title, good_item.summary)

        p = _mk_pipeline(db_path, llm_filter=fake_filter)
        p.run_cycle_with_delivery()

        # LLM должен быть вызван ровно 1 раз (для good_item), bad_item отсеян.
        assert call_count[0] == 1, f"Expected 1 LLM call, got {call_count[0]}"

    def test_garbage_skipped_state(self, tmp_path):
        """Отсеянный item: state='skipped', score=1, reason='prefilter'."""
        db_path = str(tmp_path / "test.db")
        Path(db_path).unlink(missing_ok=True)
        Storage(db_path).close()

        bad_item = self._item(
            "Sources say: major changes are coming very soon",
            summary="According to insiders, something is about to happen",
        )
        _insert_item(db_path, bad_item.url, bad_item.title, bad_item.summary)

        p = _mk_pipeline(db_path, llm_filter=FakeLLMFilter())
        p.run_cycle_with_delivery()

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT state, llm_score, llm_reason FROM items WHERE url=?",
            (bad_item.url,),
        ).fetchone()
        conn.close()
        assert row is not None, "Item should be in DB"
        state, score, reason = row
        assert state == "skipped", f"Expected state='skipped', got '{state}'"
        assert score == 1, f"Expected score=1, got {score}"
        assert reason == "prefilter", f"Expected reason='prefilter', got '{reason}'"

    def test_good_news_no_prefilter(self, tmp_path):
        """Реальная новость с фактами — НЕ отсеивается."""
        db_path = str(tmp_path / "test.db")
        Path(db_path).unlink(missing_ok=True)
        Storage(db_path).close()

        good_item = self._item(
            "Krasnodar Region: fire at industrial facility, 3 injured, evacuated",
            summary="Emergency services responded to the incident",
        )
        _insert_item(db_path, good_item.url, good_item.title, good_item.summary)

        fake_filter = FakeLLMFilter()
        call_count = [0]
        original_eval = fake_filter.evaluate_batch

        def counting_eval(items, feedback_context=None, **kwargs):
            call_count[0] += 1
            return original_eval(items, feedback_context)

        fake_filter.evaluate_batch = counting_eval

        p = _mk_pipeline(db_path, llm_filter=fake_filter)
        p.run_cycle_with_delivery()

        assert call_count[0] == 1, "Good news should go to LLM"

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT state FROM items WHERE url=?", (good_item.url,),
        ).fetchone()
        conn.close()
        assert row[0] == "delivered", f"Expected 'delivered', got '{row[0]}'"

    def test_prefilter_disabled_garbage_goes_to_llm(self, tmp_path):
        """llm.prefilter.enabled=false → все идут в LLM."""
        db_path = str(tmp_path / "test.db")
        Path(db_path).unlink(missing_ok=True)
        Storage(db_path).close()

        bad_item = self._item(
            "Celebrity wedding: Hollywood star gets married in private ceremony",
            summary="Celebrity gossip about the royal family",
        )
        _insert_item(db_path, bad_item.url, bad_item.title, bad_item.summary)

        fake_filter = FakeLLMFilter()
        call_count = [0]
        original_eval = fake_filter.evaluate_batch

        def counting_eval(items, feedback_context=None, **kwargs):
            call_count[0] += 1
            return original_eval(items, feedback_context)

        fake_filter.evaluate_batch = counting_eval

        p = _mk_pipeline(db_path, llm_filter=fake_filter,
                          extra_config={"llm": {"prefilter": {"enabled": False}}})
        p.run_cycle_with_delivery()

        # При выключенном префильтре даже мусор идёт в LLM
        assert call_count[0] == 1, "With prefilter disabled, garbage should reach LLM"

        # Но LLM его отфильтрует (score=8 → ok=True, push при нашем fake-фильтре)
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT state, llm_score FROM items WHERE url=?", (bad_item.url,),
        ).fetchone()
        conn.close()
        # FakeLLMFilter даёт score=8 (push) для всех → delivered
        assert row[0] in ("delivered", "pending"), f"Unexpected state: {row[0]}"

    def test_prefilter_then_llm_dedup(self, tmp_path, caplog):
        """
        Отсеянные префильтром НЕ попадают в LLM-дедуп.
        Не-отсеянные — попадают.
        """
        db_path = str(tmp_path / "test.db")
        Path(db_path).unlink(missing_ok=True)
        Storage(db_path).close()

        fake_filter = FakeLLMFilter()

        # Мусорный item — отсеивается префильтром
        garbage_url = "https://ex.com/garbage"
        garbage = self._item(
            title="Expert Opinion: economic outlook remains unclear",
            summary="Many analysts predict downturn without specific data",
            url=garbage_url,
        )
        _insert_item(db_path, garbage.url, garbage.title, garbage.summary)

        # Реальный item — не отсеивается
        real_url = "https://ex.com/real"
        real = self._item(
            title="Drone attack reported near Voronezh: 3 injured, evacuation ordered",
            summary="Air defense systems were active in the area",
            url=real_url,
        )
        _insert_item(db_path, real.url, real.title, real.summary)

        p = _mk_pipeline(db_path, llm_filter=fake_filter)
        p.run_cycle_with_delivery()

        # Garbage не попал в dedup
        assert garbage_url not in fake_filter._dedup_seen, (
            f"Garbage URL should NOT reach LLM dedup, got: {fake_filter._dedup_seen}"
        )
        # Real попал в dedup
        assert real_url in fake_filter._dedup_seen, (
            f"Real news URL should reach LLM dedup, got: {fake_filter._dedup_seen}"
        )
        # except-ветка _run_llm_dedup НЕ срабатывала
        assert "_run_llm_dedup: исключение" not in caplog.text, (
            "_run_llm_dedup should not raise exceptions"
        )

        p.close()


# ─────────────────────────────────────────────────────────────────
# Тесты V3: empty summary (two-phase phase-1 only, budget cut)
# ─────────────────────────────────────────────────────────────────

class TestEmptySummaryV3:
    """V3 two-phase: phase-1 only даёт пустой summary → не должен ломать цикл.

    Тестируем через прямой вызов _process_batch с моком LLM,
    чтобы избежать зависимости от delivery/send_batch.
    """

    def _item(self, title, summary="RSS summary text", url=None):
        if url is None:
            import hashlib
            url = "https://ex.com/" + hashlib.md5(title.encode()).hexdigest()[:8]
        return _make_item(url=url, title=title, summary=summary)

    def test_empty_summary_score_low_skipped_with_rss_fallback(self, tmp_path):
        """
        Item score=2 (digest_threshold=5), summary="" (phase-1 only) →
        цикл проходит, item → skipped с RSS-fallback в llm_summary.
        """
        db_path = str(tmp_path / "test.db")
        Path(db_path).unlink(missing_ok=True)
        Storage(db_path).close()

        item = self._item(
            title="Some News Event",
            summary="RSS fallback text here",
            url="https://ex.com/empty_low",
        )
        _insert_item(
            db_path, item.url, item.title, item.summary,
            llm_score=2, llm_summary="", llm_reason="low score",
        )

        fake_filter = FakeLLMFilterEmptySummary()
        config = {
            "sources": [],
            "database": {"path": db_path},
            "telegram": {"token": "", "chat_id": ""},
            "poll_interval_minutes": 15,
            "digest": {"enabled": True, "max_items": 100, "max_items_per_message": 20},
            "llm": {"enabled": True, "api_key": "fake"},
        }
        p = NewsPipeline(config)
        p.llm_filter = fake_filter
        p.storage.close()
        p.storage = Storage(db_path)

        from unittest.mock import MagicMock
        delivery = MagicMock(spec=__import__(
            "newsbot.delivery.telegram", fromlist=["TelegramDelivery"]
        ).TelegramDelivery)
        delivery.send_batch.return_value = ([], [])  # no pushes, no failures
        delivery.send_text.return_value = True
        p.delivery = delivery

        # Запускаем _process_batch напрямую — без полного run_cycle_with_delivery
        # (которое требует working delivery для digest)
        result = p._process_batch([item])

        # Проверяем что llm_summary записан с RSS-fallback
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT state, llm_score, llm_summary FROM items WHERE url=?",
            (item.url,),
        ).fetchone()
        conn.close()

        assert row is not None, "Item should be in DB"
        state, score, stored_summary = row
        assert state == "skipped", f"Expected state='skipped', got '{state}'"
        # V3-fix: llm_summary записан как RSS-fallback (не пустой)
        assert stored_summary == "RSS fallback text here", (
            f"Expected RSS fallback in DB, got {stored_summary!r}"
        )
        # push не вызван (item отфильтрован)
        delivery.send_text.assert_not_called()

        p.close()

    def test_empty_summary_score_high_deferred_with_rss_fallback(self, tmp_path):
        """
        Item score=8 (>= digest_threshold), summary="" (phase-2 пропущен) →
        item → deferred с RSS-fallback в llm_summary (не пустой).
        """
        db_path = str(tmp_path / "test.db")
        Path(db_path).unlink(missing_ok=True)
        Storage(db_path).close()

        item = self._item(
            title="Breaking News Event",
            summary="Original RSS summary text for this item",
            url="https://ex.com/empty_high",
        )
        _insert_item(
            db_path, item.url, item.title, item.summary,
            llm_score=8, llm_summary="", llm_reason="high score",
        )

        fake_filter = FakeLLMFilterHighScore()
        config = {
            "sources": [],
            "database": {"path": db_path},
            "telegram": {"token": "", "chat_id": ""},
            "poll_interval_minutes": 15,
            "digest": {
                "enabled": True, "max_items": 100, "max_items_per_message": 20,
                "max_per_location": 5,
            },
            "llm": {"enabled": True, "api_key": "fake"},
        }
        p = NewsPipeline(config)
        p.llm_filter = fake_filter
        p.storage.close()
        p.storage = Storage(db_path)

        from unittest.mock import MagicMock
        delivery = MagicMock(spec=__import__(
            "newsbot.delivery.telegram", fromlist=["TelegramDelivery"]
        ).TelegramDelivery)
        delivery.send_batch.return_value = ([], [])  # no pushes
        delivery.send_text.return_value = True
        p.delivery = delivery

        result = p._process_batch([item])

        # Проверяем llm_summary в БД
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT state, llm_summary FROM items WHERE url=?",
            (item.url,),
        ).fetchone()
        conn.close()

        assert row is not None, "Item should be in DB"
        state, stored_summary = row
        # Item попал в deferred
        assert state == "deferred", f"Expected state='deferred', got '{state}'"
        # V3-fix: RSS-fallback (не пустой)
        assert stored_summary, f"Expected non-empty RSS fallback, got {stored_summary!r}"
        assert stored_summary == "Original RSS summary text for this item"

        p.close()

    # ── Minor #3: интеграционные тесты префильтра ──────────────────────────

    def test_all_garbage_no_llm_call(self, tmp_path):
        """
        Все items — мусор → _process_batch возвращает [], state=skipped,
        LLM НЕ вызван (call_count == 0).
        """
        db_path = str(tmp_path / "test.db")
        Path(db_path).unlink(missing_ok=True)
        Storage(db_path).close()

        fake_filter = FakeLLMFilter()
        call_count = [0]
        original_eval = fake_filter.evaluate_batch

        def counting_eval(items, feedback_context=None, **kwargs):
            call_count[0] += 1
            return original_eval(items, feedback_context)

        fake_filter.evaluate_batch = counting_eval

        # 3 мусорных items (БЕЗ цифр — цифра в title ложно определяется как "факт")
        garbage = [
            ("Expert Opinion: economy will face pressure", "Many analysts predict downturn"),
            ("Sources Say: changes are coming soon", "According to insiders, something might happen"),
            ("Daily Fuel Price Report across markets", "Fuel prices fell across regions"),
        ]
        for title, summary in garbage:
            item = self._item(title=title, summary=summary)
            _insert_item(db_path, item.url, item.title, item.summary)

        p = _mk_pipeline(db_path, llm_filter=fake_filter)
        p.run_cycle_with_delivery()

        # LLM не вызван (все отсеяны префильтром)
        assert call_count[0] == 0, f"Expected 0 LLM calls, got {call_count[0]}"
        # Все — skipped
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT state FROM items").fetchall()
        conn.close()
        assert all(r[0] == "skipped" for r in rows), (
            f"All items should be skipped, got: {[r[0] for r in rows]}"
        )
        p.close()

    def test_scored_item_not_prefiltered(self, tmp_path):
        """
        Item с llm_score=8 (уже оценён) — НЕ префильтруется
        (scored items проходят мимо _prefilter_item).
        """
        db_path = str(tmp_path / "test.db")
        Path(db_path).unlink(missing_ok=True)
        Storage(db_path).close()

        fake_filter = FakeLLMFilter()
        call_count = [0]
        original_eval = fake_filter.evaluate_batch

        def counting_eval(items, feedback_context=None, **kwargs):
            call_count[0] += 1
            return original_eval(items, feedback_context)

        fake_filter.evaluate_batch = counting_eval

        # Мусорный title, но llm_score уже сохранён (retry/recovered)
        item = self._item(
            title="Expert Opinion: market outlook looks grim",
            summary="Many analysts believe things will worsen",
            url="https://ex.com/scored",
        )
        _insert_item(
            db_path, item.url, item.title, item.summary,
            llm_score=8,  # уже оценён
        )

        p = _mk_pipeline(db_path, llm_filter=fake_filter)
        p.run_cycle_with_delivery()

        # LLM не вызван для scored item (retry без LLM-оценки)
        # LLM может быть вызван для других fresh items, но их нет
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT state, llm_score FROM items WHERE url=?",
            (item.url,),
        ).fetchone()
        conn.close()
        assert row is not None, "Item should be in DB"
        state, score = row
        assert state != "skipped", f"Scored item should NOT be skipped, got state={state}"
        assert score == 8, f"Original score should be preserved, got {score}"
        p.close()


class FakeLLMFilterHighScore:
    """Fake-фильтр: score=8, пустой summary (имитация phase2-budget-cut).

    push_threshold=9 — чтобы score=8 маршрутизировался в 'digest', не в 'push'.
    """
    def __init__(self):
        self.push_threshold = 9
        self.digest_threshold = 5
        self.push_urgent_only = False

    def evaluate_batch(self, items, feedback_context=None):
        return [
            FilterResult(ok=True, score=8, summary="", reason="high score",
                         urgent=False, location="", title_en=None)
            for i in items
        ]

    def get_balance(self):
        class _Bal:
            is_available = True
            total_balance = 100.0
            used_today = 0.0
            remaining = 100.0
            currency = "USD"
        return _Bal()


