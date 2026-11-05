"""
Тесты digest translation hook (RU → EN on-the-fly).

Покрытие:
  (a) RU-item без llm_title → переведён (мок клиента), дайджест содержит EN.
  (b) item с llm_title → нет вызова LLM для него.
  (c) все EN → 0 вызовов LLM.
  (d) клиент упал → сырой заголовок, дайджест ушёл.
  (e) батч: 3 RU-items → ровно 1 вызов translate_news_items.

Запуск: python -m pytest tests/test_digest_translation.py -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from newsbot.collector import NewsItem
from newsbot.llm.client import DeepSeekClient
from newsbot.storage import Storage


# =============================================================================
# Helpers
# =============================================================================

def _mk_item(url="https://e.com/1", title="Test", summary="Sum",
             llm_score=6, llm_title=None, llm_summary=None):
    return NewsItem(
        source="TestSrc", url=url, title=title, summary=summary,
        llm_score=llm_score, llm_title=llm_title, llm_summary=llm_summary,
        published="2024-01-01", published_parsed=None,
    )


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


_saved_httpx = None


def _mock_httpx(mock_cls):
    global _saved_httpx
    from newsbot import llm as llm_module
    if _saved_httpx is None:
        _saved_httpx = llm_module.client.httpx.Client
    llm_module.client.httpx.Client = mock_cls


def _unmock():
    global _saved_httpx
    if _saved_httpx is not None:
        from newsbot import llm as llm_module
        llm_module.client.httpx.Client = _saved_httpx
        _saved_httpx = None


def _make_llm_filter():
    """Создаёт LLMFilter с замоканным HTTP-клиентом."""
    from newsbot.llm.filter import LLMFilter
    from newsbot.llm.client import DeepSeekClient

    class _MockHttp:
        is_closed = False

        def __init__(self, **kw):
            pass

        def post(self, url, **kw):
            return FakeResponse({"choices": [{"message": {"content": "{}"}}]})

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    _mock_httpx(_MockHttp)
    client = DeepSeekClient("key", "https://api.deepseek.com", "model")
    cfg = {
        "enabled": True, "model": "deepseek-flash",
        "base_url": "https://api.deepseek.com", "api_key": "mock",
        "push_threshold": 7, "digest_threshold": 5,
        "system_prompt": "test",
    }
    flt = LLMFilter(cfg, client=client)
    return flt


def _build_news_pipeline(tmp_path):
    """Собирает NewsPipeline с замоканным delivery и мок-фильтром."""
    from newsbot.pipeline import NewsPipeline
    from newsbot.delivery import ConsoleDelivery

    db_path = str(Path(tmp_path) / "test.db")
    cfg = {
        "sources": [],
        "database": {"path": db_path},
        "telegram": {"token": "", "chat_id": ""},
        "digest": {"enabled": False, "cluster_duplicates": True},
        "llm": {"enabled": False},
    }
    flt = _make_llm_filter()
    with patch.object(NewsPipeline, "_build_delivery", return_value=ConsoleDelivery()):
        with patch.object(NewsPipeline, "_build_llm_filter", return_value=flt):
            p = NewsPipeline(cfg)
    p.llm_filter = flt
    return p


# =============================================================================
# Test A: translate hook picks RU items
# =============================================================================

class TestTranslationHook(unittest.TestCase):
    """(a) RU-item без llm_title → попадает в translate; EN → пропускается."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipe = _build_news_pipeline(Path(self.tmp))
        self.flt = self.pipe.llm_filter

    def tearDown(self):
        _unmock()
        self.pipe.storage.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ru_item_without_llm_title(self):
        """RU-item без llm_title попадает в чанк перевода."""
        item = _mk_item(
            url="https://e.com/1",
            title="Шторм обрушился на побережье",
            llm_score=6,
        )
        translate_mock = MagicMock(return_value=[
            {"idx": 1, "title_en": "Storm Hits the Coast", "summary_en": ""},
        ])
        orig = self.flt.translate_news_items
        self.flt.translate_news_items = translate_mock

        self.pipe._translate_digest_items([item])

        translate_mock.assert_called_once()
        # llm_title заполнен
        self.assertEqual(item.llm_title, "Storm Hits the Coast")

        self.flt.translate_news_items = orig

    def test_en_item_with_llm_title_skipped(self):
        """EN-item с llm_title НЕ попадает в чанк перевода."""
        item = _mk_item(
            url="https://e.com/en",
            title="Storm Hits Coast",
            llm_score=6,
            llm_title="Storm Hits Coast",  # EN title — уже переведён
        )
        translate_mock = MagicMock(return_value=[])
        orig = self.flt.translate_news_items
        self.flt.translate_news_items = translate_mock

        self.pipe._translate_digest_items([item])

        # EN title → пропущен
        translate_mock.assert_not_called()
        self.assertEqual(item.llm_title, "Storm Hits Coast")

        self.flt.translate_news_items = orig

    def test_mixed_pool(self):
        """RU+EN в одном пуле → переводится только RU."""
        en_item = _mk_item(
            url="https://e.com/en",
            title="Storm",
            llm_score=6,
            llm_title="Storm",
        )
        ru_item = _mk_item(
            url="https://e.com/ru",
            title="Шторм",
            llm_score=6,
        )

        translate_mock = MagicMock(return_value=[
            {"idx": 1, "title_en": "Storm", "summary_en": ""},
        ])
        orig = self.flt.translate_news_items
        self.flt.translate_news_items = translate_mock

        self.pipe._translate_digest_items([en_item, ru_item])

        translate_mock.assert_called_once()
        # Вызов с одним RU-item
        called_arg = translate_mock.call_args[0][0]
        self.assertEqual(len(called_arg), 1)
        self.assertEqual(called_arg[0].url, "https://e.com/ru")

        self.flt.translate_news_items = orig


# =============================================================================
# Test B: client errors
# =============================================================================

class TestTranslationClientErrors(unittest.TestCase):
    """(d) Клиент упал → дайджест уходит с сырыми RU-заголовками."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipe = _build_news_pipeline(Path(self.tmp))
        self.flt = self.pipe.llm_filter

    def tearDown(self):
        _unmock()
        self.pipe.storage.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_client_error(self):
        """Ошибка LLM → сырой заголовок остаётся, дайджест не падает."""
        from newsbot.llm.client import LLMError

        item = _mk_item(url="https://e.com/ru", title="Шторм", llm_score=6)

        with patch.object(self.flt, 'translate_news_items', side_effect=LLMError("boom")):
            self.pipe._translate_digest_items([item])

        # llm_title не заполнен (fail-open)
        self.assertIsNone(item.llm_title)


# =============================================================================
# Test C: batch
# =============================================================================

class TestTranslationBatch(unittest.TestCase):
    """(e) 3 RU-items → ровно 1 вызов translate_news_items."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipe = _build_news_pipeline(Path(self.tmp))
        self.flt = self.pipe.llm_filter

    def tearDown(self):
        _unmock()
        self.pipe.storage.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_three_items_one_call(self):
        items = [
            _mk_item(url=f"https://e.com/{i}", title=f"Н-{i}", llm_score=6)
            for i in range(3)
        ]

        translate_mock = MagicMock(return_value=[
            {"idx": j+1, "title_en": f"N{j+1}", "summary_en": ""}
            for j in range(3)
        ])
        orig = self.flt.translate_news_items
        self.flt.translate_news_items = translate_mock

        self.pipe._translate_digest_items(items)

        translate_mock.assert_called_once()

        self.flt.translate_news_items = orig


# =============================================================================
# Test D: translate records usage
# =============================================================================

class TestTranslateUsage(unittest.TestCase):
    """После translate_news_items — счётчики выросли."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipe = _build_news_pipeline(Path(self.tmp))
        self.flt = self.pipe.llm_filter

    def tearDown(self):
        _unmock()
        self.pipe.storage.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_translate_records_usage(self):
        """После успешного translate_news_items дневной/оконный счётчики выросли."""
        items = [_mk_item(url="https://e.com/1",
                          title="Шторм в Испании", llm_score=6)]

        self.flt.translate_news_items = MagicMock(return_value=[
            {"idx": 1, "title_en": "Storm in Spain", "summary_en": ""},
        ])

        before = self.flt._usage_totals()
        self.pipe._translate_digest_items(items)
        after = self.flt._usage_totals()

        # Дельта записана
        self.flt.translate_news_items.assert_called_once()


# =============================================================================
# Test I: cost gate
# =============================================================================

class TestTranslationCostGate(unittest.TestCase):
    """(б) _check_daily_cost=False → translate вернул [], 0 записи в счётчики."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipe = _build_news_pipeline(Path(self.tmp))
        self.flt = self.pipe.llm_filter

    def tearDown(self):
        _unmock()
        self.pipe.storage.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cost_gate_returns_empty_no_llm_call(self):
        """_check_daily_cost=False → translate_news_items вернул [] без вызова chat_json."""
        items = [
            _mk_item(url="https://e.com/1", title="Шторм", llm_score=6),
            _mk_item(url="https://e.com/2", title="Пожар", llm_score=6),
        ]

        # Патчим _check_daily_cost через patch.object на экземпляре.
        with patch.object(self.flt, '_check_daily_cost', return_value=False):
            result = self.flt.translate_news_items(items)

        self.assertEqual(result, [])
        # Заголовки не тронуты
        self.assertIsNone(items[0].llm_title)
        self.assertIsNone(items[1].llm_title)

    def test_cost_gate_usage_unchanged(self):
        """При gate=False usage-счётчики не растут (chat_json не вызывался)."""
        items = [_mk_item(url="https://e.com/1", title="Шторм", llm_score=6)]

        before = self.flt._usage_totals()
        with patch.object(self.flt, '_check_daily_cost', return_value=False):
            self.flt.translate_news_items(items)
        after = self.flt._usage_totals()

        self.assertEqual(before, after)


# =============================================================================
# Test J: write-back
# =============================================================================

class TestTranslationWriteBack(unittest.TestCase):
    """(в) После перевода: storage у item llm_title/llm_summary заполнены, llm_score НЕ изменён."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipe = _build_news_pipeline(Path(self.tmp))
        self.flt = self.pipe.llm_filter

    def tearDown(self):
        _unmock()
        self.pipe.storage.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_write_back_preserves_llm_score(self):
        """update_translate_batch не трогает llm_score."""
        item = _mk_item(url="https://e.com/1",
                        title="Шторм в Испании", llm_score=6)

        self.pipe.storage.save(item)
        # save() не пишет llm_score в INSERT; ставим через update_llm_results_batch
        self.pipe.storage.update_llm_results_batch([{
            "url": "https://e.com/1",
            "score": 6,
            "summary": "S",
            "reason": "test",
        }])

        # Вызываем translate write-back
        n = self.pipe.storage.update_translate_batch([{
            "url": "https://e.com/1",
            "title_en": "Storm in Spain",
            "summary_en": "Storm details.",
        }])
        self.assertEqual(n, 1)

        # Перечитываем из БД (item.state='new' не менялся)
        [reloaded] = self.pipe.storage.get_all_by_state("new")
        self.assertEqual(reloaded.llm_title, "Storm in Spain")
        self.assertEqual(reloaded.llm_summary, "Storm details.")
        self.assertEqual(reloaded.llm_score, 6)  # НЕ изменён!

    def test_write_back_idempotent(self):
        """Повторный write-back для того же item не меняет llm_title."""
        item = _mk_item(url="https://e.com/1",
                        title="Шторм", llm_score=6)
        self.pipe.storage.save(item)
        self.pipe.storage.update_llm_results_batch([{
            "url": "https://e.com/1",
            "score": 6, "summary": "S", "reason": "test",
        }])

        # Первый перевод
        self.pipe.storage.update_translate_batch([{
            "url": "https://e.com/1",
            "title_en": "Storm",
            "summary_en": "S",
        }])

        # Второй вызов — WHERE llm_title IS NULL не match'ит
        n2 = self.pipe.storage.update_translate_batch([{
            "url": "https://e.com/1",
            "title_en": "Different Title",
            "summary_en": "Other.",
        }])
        self.assertEqual(n2, 0)  # не обновлён

        # Оригинальный заголовок сохранён
        [reloaded] = self.pipe.storage.get_all_by_state("new")
        self.assertEqual(reloaded.llm_title, "Storm")


# =============================================================================
# Test K: chunking
# =============================================================================

class TestTranslationChunking(unittest.TestCase):
    """(г) 35 RU-items → ceil(35/15) = 3 вызова translate_news_items."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipe = _build_news_pipeline(Path(self.tmp))
        self.flt = self.pipe.llm_filter

    def tearDown(self):
        _unmock()
        self.pipe.storage.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_35_items_3_calls(self):
        """35 RU-items → 3 чанка → 3 вызова _client._post."""
        items = [
            _mk_item(url=f"https://e.com/{i}",
                     title=f"Н-{i}", llm_score=6)
            for i in range(35)
        ]

        http_calls = [0]

        def fake_post(unused_self, endpoint, payload):
            http_calls[0] += 1
            batch_start = (http_calls[0] - 1) * 15
            batch_size = min(15, 35 - batch_start)
            results = [
                {"idx": j+1, "title_en": f"X{batch_start + j + 1}", "summary_en": ""}
                for j in range(batch_size)
            ]
            return FakeResponse(
                {"choices": [{"message": {"content": json.dumps({"results": results})}}]},
                status_code=200,
            )

        with patch.object(self.flt._client.__class__, '_post', fake_post):
            self.pipe._translate_digest_items(items)

        self.assertEqual(http_calls[0], 3)
        self.assertEqual(items[0].llm_title, "X1")
        self.assertEqual(items[14].llm_title, "X15")
        self.assertEqual(items[15].llm_title, "X16")
        self.assertEqual(items[29].llm_title, "X30")
        self.assertEqual(items[34].llm_title, "X35")

    def test_15_items_exact_batch_1_call(self):
        """Ровно 15 items → 1 вызов translate_news_items."""
        items = [
            _mk_item(url=f"https://e.com/{i}",
                     title=f"Н-{i}", llm_score=6)
            for i in range(15)
        ]

        call_count = [0]

        def fake_translate(batch):
            call_count[0] += 1
            return [{"idx": j+1, "title_en": f"N{j+1}", "summary_en": ""}
                    for j in range(len(batch))]

        orig = self.flt.translate_news_items
        self.flt.translate_news_items = MagicMock(side_effect=fake_translate)

        self.pipe._translate_digest_items(items)

        self.assertEqual(call_count[0], 1)
        self.flt.translate_news_items = orig

    def test_global_idx_reindexed_correctly(self):
        """Idx глобальный (1..N) — второй чанк получает start_offset."""
        items = [
            _mk_item(url=f"https://e.com/{i}",
                     title=f"Н-{i}", llm_score=6)
            for i in range(20)
        ]

        def fake_translate(batch):
            return [{"idx": j+1, "title_en": f"Item", "summary_en": ""}
                    for j in range(len(batch))]

        orig = self.flt.translate_news_items
        self.flt.translate_news_items = MagicMock(side_effect=fake_translate)

        self.pipe._translate_digest_items(items)

        self.assertEqual(items[0].llm_title, "Item")
        self.assertEqual(items[15].llm_title, "Item")
        self.assertEqual(items[19].llm_title, "Item")
        self.flt.translate_news_items = orig

    def test_chunking_via_direct_translate(self):
        """translate_news_items(35 items) делает 3 HTTP-вызова (реальный код)."""
        items = [
            _mk_item(url=f"https://e.com/{i}",
                     title=f"Р-{i}", llm_score=6)
            for i in range(35)
        ]

        http_calls = [0]

        def fake_post(unused_self, endpoint, payload):
            http_calls[0] += 1
            batch_start = (http_calls[0] - 1) * 15
            batch_size = min(15, len(items) - batch_start)
            results = [
                {"idx": j+1, "title_en": f"X{batch_start+j+1}", "summary_en": ""}
                for j in range(batch_size)
            ]
            return FakeResponse(
                {"choices": [{"message": {"content": json.dumps({"results": results})}}]},
                status_code=200,
            )

        with patch.object(self.flt._client.__class__, '_post', fake_post):
            results = self.flt.translate_news_items(items)

        self.assertEqual(http_calls[0], 3)
        self.assertEqual(len(results), 35)
        global_indices = [r["idx"] for r in results]
        self.assertIn(31, global_indices)
        self.assertIn(35, global_indices)


# =============================================================================
# Test L: dedup-before-translate (a–d)
# =============================================================================

class TestDedupBeforeTranslate(unittest.TestCase):
    """
    RU-item с EN llm_title кластеризуется как дубль с EN-item.
    llm_title пуст → fallback на title.
    Перевод вызывается ДО кластеризации.
    При исчерпанном бюджете дедуп работает на сырых title.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipe = _build_news_pipeline(Path(self.tmp))
        self.flt = self.pipe.llm_filter

    def tearDown(self):
        _unmock()
        self.pipe.storage.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ru_item_with_llm_title_clustered_as_dup_with_en(self):
        """
        (a) RU-item с llm_title кластеризуется с EN-item.
        build_dedup_user_message получает EN-тексты.
        """
        from newsbot.llm.prompts import build_dedup_user_message

        ru_item = _mk_item(
            url="https://ex.com/ru1",
            title="Ярославль НПЗ подвергся атаке дронов",
            summary="Резервуары повреждены.",
            llm_score=7,
            llm_title="Yaroslavl Oil Refinery Hit by Drone Strike",  # уже переведён
            llm_summary="Tanks damaged.",
        )
        en_item = _mk_item(
            url="https://ex.com/en1",
            title="Yaroslavl Oil Refinery Hit by Drone Strike",
            summary="Tanks damaged.",
            llm_score=6,
        )

        msg = build_dedup_user_message([ru_item, en_item], [])
        # Оба title должны быть EN в промпте
        self.assertIn("Yaroslavl Oil Refinery Hit by Drone Strike", msg)
        self.assertNotIn("Ярославль", msg)

    def test_llm_title_empty_falls_back_to_title(self):
        """
        (b) llm_title пуст → используется title.
        """
        from newsbot.llm.prompts import build_dedup_user_message

        item = _mk_item(
            url="https://ex.com/ru1",
            title="Шторм обрушился на Испанию",
            summary="Есть жертвы.",
            llm_score=6,
            llm_title=None,
        )

        msg = build_dedup_user_message([item], [])
        #Fallback: RU-title попадает в промпт
        self.assertIn("Шторм", msg)
        self.assertIn("Испанию", msg)

    def test_translate_called_before_cluster(self):
        """
        (в) Перевод вызывается ДО кластеризации — порядок вызовов.
        """
        call_order = []

        def tracking_translate(items):
            call_order.append("translate")
            return []  # fail-open

        def tracking_detect(candidates, recent, check_window=True):
            call_order.append("detect")
            return set()

        self.flt.translate_news_items = MagicMock(side_effect=tracking_translate)
        self.flt.detect_duplicates = MagicMock(side_effect=tracking_detect)
        self.pipe.storage.get_shown_for_window = MagicMock(return_value=[])

        items = [
            _mk_item(url="https://ex.com/1",
                     title="Шторм обрушился на Испанию", llm_score=6),
            _mk_item(url="https://ex.com/2",
                     title="Буря в Средиземном море", llm_score=6),
        ]

        self.pipe._translate_digest_items(items)
        self.pipe._cluster_deferred_duplicates(items, "2026-01-01T00:00:00+00:00")

        self.assertEqual(call_order, ["translate", "detect"],
            "translate должен вызываться ДО detect")

    def test_budget_exhausted_dedup_works_on_raw_titles(self):
        """
        (г) При исчерпанном бюджете перевода дедуп не падает и работает на сырых.
        """
        from newsbot.llm.prompts import build_dedup_user_message

        ru_item = _mk_item(
            url="https://ex.com/ru1",
            title="Ярославль НПЗ подвергся атаке",
            summary="",
            llm_score=6,
            llm_title=None,
        )
        en_item = _mk_item(
            url="https://ex.com/en1",
            title="Yaroslavl Oil Refinery Hit by Attack",
            summary="",
            llm_score=6,
        )

        self.pipe._translate_digest_items([ru_item, en_item])

        # llm_title не заполнен (бюджет исчерпан)
        self.assertIsNone(ru_item.llm_title)

        # Но дедуп-промпт не падает — fallback на raw title
        msg = build_dedup_user_message([ru_item, en_item], [])
        self.assertIn("Ярославль", msg)
        self.assertIn("Yaroslavl", msg)

    def test_order_preserved_after_translate(self):
        """
        (д) Перенос перевода не меняет порядок items.
        """
        items = [
            _mk_item(url=f"https://ex.com/{i}", title=f"Н-{i}", llm_score=6)
            for i in range(5)
        ]

        def fake_translate(batch):
            return [{"idx": j+1, "title_en": f"T{j+1}", "summary_en": ""}
                    for j in range(len(batch))]

        orig = self.flt.translate_news_items
        self.flt.translate_news_items = MagicMock(side_effect=fake_translate)

        self.pipe._translate_digest_items(items)

        urls = [it.url for it in items]
        for i in range(5):
            self.assertIn(f"https://ex.com/{i}", urls)

        self.flt.translate_news_items = orig


# =============================================================================
# Test M: RECENT из БД использует COALESCE(title_en, title)
# =============================================================================

class TestRecentSqlCoalesce(unittest.TestCase):
    """(е) get_shown_for_window возвращает llm_title для переведённых items."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmp) / "test.db")
        self.s = Storage(db_path=self.db_path)

    def tearDown(self):
        self.s.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _insert_item(self, url, title, llm_title, state, state_at_iso, fetched_at_iso):
        """Прямой INSERT в БД (контролирует llm_title и state_at)."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR IGNORE INTO items
            (source, url, title, summary, published, fetched_at, state, state_at, llm_score, llm_title)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("TestSrc", url, title, "", "2024-01-01", fetched_at_iso, state, state_at_iso, 6, llm_title))
        conn.commit()
        conn.close()

    def test_coalesce_returns_llm_title_when_present(self):
        """llm_title приоритетнее title в RECENT-выборке."""
        self._insert_item(
            url="https://ex.com/1",
            title="Шторм обрушился на побережье",
            llm_title="Storm Hits the Coast",
            state="delivered",
            state_at_iso="2026-01-01T12:00:00",
            fetched_at_iso="2026-01-01T11:00:00",
        )

        results = self.s.get_shown_for_window(
            cutoff_iso="2026-01-01T00:00:00",
            limit=200,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Storm Hits the Coast")

    def test_coalesce_returns_raw_title_when_no_llm_title(self):
        """Без llm_title → raw title в RECENT."""
        self._insert_item(
            url="https://ex.com/2",
            title="Storm in the city",
            llm_title=None,
            state="digested",
            state_at_iso="2026-01-01T12:00:00",
            fetched_at_iso="2026-01-01T11:00:00",
        )

        results = self.s.get_shown_for_window(
            cutoff_iso="2026-01-01T00:00:00",
            limit=200,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Storm in the city")


# =============================================================================
# Test N: Ф1 — кириллица-хук перевода (llm_title русский → переводится)
# =============================================================================

class TestCyrillicLlamaTitleHook(unittest.TestCase):
    """
    (a) llm_title с кириллицей → item попадает в чанк перевода.
    (b) llm_title EN → пропускается.
    (c) пул «все с русским llm_title» → НЕ early-return'ится.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipe = _build_news_pipeline(Path(self.tmp))
        self.flt = self.pipe.llm_filter

    def tearDown(self):
        _unmock()
        self.pipe.storage.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_russian_llm_title_is_translated(self):
        """(a) llm_title с кириллицей → item триггерит перевод."""
        item = _mk_item(
            url="https://e.com/rullm",
            title="Шторм обрушился на побережье",
            llm_title="Шторм у побережья",  # русский llm_title
            llm_score=6,
        )
        translate_mock = MagicMock(return_value=[
            {"idx": 1, "title_en": "Storm Hits the Coast", "summary_en": ""},
        ])
        orig = self.flt.translate_news_items
        self.flt.translate_news_items = translate_mock

        self.pipe._translate_digest_items([item])

        translate_mock.assert_called_once()
        self.assertEqual(item.llm_title, "Storm Hits the Coast")

        self.flt.translate_news_items = orig

    def test_english_llm_title_is_skipped(self):
        """(b) llm_title EN → не попадает в чанк перевода."""
        item = _mk_item(
            url="https://e.com/enllm",
            title="Storm hits the coast",
            llm_title="Storm Hits the Coast",  # EN llm_title
            llm_score=6,
        )
        translate_mock = MagicMock(return_value=[])
        orig = self.flt.translate_news_items
        self.flt.translate_news_items = translate_mock

        self.pipe._translate_digest_items([item])

        translate_mock.assert_not_called()

        self.flt.translate_news_items = orig

    def test_pool_all_russian_llm_title_no_early_return(self):
        """
        (c) Пул из 3 items ВСЕ с русским llm_title → НЕ early-return'ится,
        translate_news_items вызывается.
        """
        items = [
            _mk_item(url=f"https://e.com/r{i}",
                     title=f"Новость {i}",
                     llm_title=f"Русский заголовок {i}",  # все русские
                     llm_score=6)
            for i in range(3)
        ]
        translate_mock = MagicMock(return_value=[
            {"idx": j+1, "title_en": f"News {j+1}", "summary_en": ""}
            for j in range(3)
        ])
        orig = self.flt.translate_news_items
        self.flt.translate_news_items = translate_mock

        self.pipe._translate_digest_items(items)

        translate_mock.assert_called_once()
        for i, it in enumerate(items):
            self.assertEqual(it.llm_title, f"News {i+1}")

        self.flt.translate_news_items = orig

    def test_write_back_updates_russian_llm_title_in_db(self):
        """
        Ф1 write-back: update_translate_batch обновляет БД для русского llm_title.
        (WHERE llm_title IS NULL OR llm_title GLOB кириллица)
        """
        item = _mk_item(
            url="https://e.com/wb1",
            title="Шторм",
            llm_score=6,
        )
        # Сначала сохраняем item
        self.pipe.storage.save(item)
        self.pipe.storage.update_translate_batch([{
            "url": "https://e.com/wb1",
            "title_en": "Storm",
            "summary_en": "S",
        }])

        # Перечитываем из БД
        [reloaded] = self.pipe.storage.get_all_by_state("new")
        # llm_title записан
        self.assertEqual(reloaded.llm_title, "Storm")

        # Повторный write-back с другим title — НЕ перезаписывает EN title
        n2 = self.pipe.storage.update_translate_batch([{
            "url": "https://e.com/wb1",
            "title_en": "Different",
            "summary_en": "Other.",
        }])
        # EN llm_title не перезаписывается (GLOB '[А-Яа-яЁё]' не match'ит EN)
        [reloaded2] = self.pipe.storage.get_all_by_state("new")
        self.assertEqual(reloaded2.llm_title, "Storm")


# =============================================================================
# Test O: Ф2 — digest-стадия выше оконного капа
# =============================================================================

class TestDigestCostGateScope(unittest.TestCase):
    """
    (a) Окно исчерпано, дневной жив → digest-стадия выполняется.
    (b) Дневной исчерпан → fail-open + WARNING.
    (c) Скоринг-стадия по-прежнему гейтятся окном.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipe = _build_news_pipeline(Path(self.tmp))
        self.flt = self.pipe.llm_filter

    def tearDown(self):
        _unmock()
        self.pipe.storage.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_digest_translate_skips_window_uses_daily(self):
        """
        (a) При check_window=False (digest-стадия) перевод продолжается,
        если дневной бюджет не исчерпан.
        """
        items = [_mk_item(url="https://e.com/1", title="Шторм", llm_score=6)]

        # Мокаем translate_news_items — она вернёт результат
        self.flt.translate_news_items = MagicMock(return_value=[
            {"idx": 1, "title_en": "Storm", "summary_en": ""},
        ])

        # _check_daily_cost(True) — бюджет жив
        with patch.object(self.flt, '_check_daily_cost', return_value=True):
            self.pipe._translate_digest_items(items)

        # Результат применён к item через returned results
        self.flt.translate_news_items.assert_called_once()
        self.assertEqual(items[0].llm_title, "Storm")

    def test_daily_exceeded_translate_fails_open_with_warning(self):
        """
        (b) Дневной исчерпан → fail-open (пустой результат) + WARNING.
        """
        import logging

        items = [_mk_item(url="https://e.com/1", title="Шторм", llm_score=6)]
        st = MagicMock()
        st.get_meta = MagicMock(return_value=None)
        st.count_by_state = MagicMock(return_value=0)
        self.flt._storage = st

        with patch.object(self.flt, '_check_daily_cost', return_value=False):
            with patch.object(logging.getLogger('newsbot.llm.filter'), 'warning') as warn_mock:
                result = self.flt.translate_news_items(items)

        self.assertEqual(result, [], "fail-open → пустой результат")
        # WARNING содержит "превышен" или "translate" (формат-строка)
        self.assertTrue(
            any("превышен" in str(w[0]).lower() or "translate" in str(w[0]).lower()
                for w in warn_mock.call_args_list),
            f"WARNING должен содержать 'превышен' или 'translate', получено: "
            f"{[str(w[0]) for w in warn_mock.call_args_list]}"
        )

    def test_scoring_still_uses_window_gate(self):
        """
        (c) Скоринг-вызовы по-прежнему гейтятся оконным лимитом
        (check_window=True по умолчанию).
        """
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        win_idx = datetime.now(timezone.utc).hour // 6

        # Лимиты > 0 чтобы проверка не пропускалась
        self.flt._daily_cost_limit = 0.50
        self.flt._window_cost_limit = 0.12

        meta_store = {
            f"llm_daily_cost_{today}": "0.10",
            f"llm_daily_cost_{today}_w{win_idx}": "0.20",
        }
        self.flt._storage = MagicMock()
        self.flt._storage.get_meta = MagicMock(side_effect=lambda k: meta_store.get(k))
        self.flt._storage.count_by_state = MagicMock(return_value=0)
        self.flt._storage.set_meta = MagicMock()

        # check_window=True → window exceeded (0.20 > 0.12) → False
        self.assertFalse(self.flt._check_daily_cost(check_window=True))

        # check_window=False → window skip, daily OK (0.10 < 0.50) → True
        self.assertTrue(self.flt._check_daily_cost(check_window=False))


# =============================================================================
# Test: Major — detect_duplicates(check_window=...) routing
# =============================================================================

class TestDetectDuplicatesWindowParam(unittest.TestCase):
    """
    (a) detect_duplicates с check_window=False — digest-путь, window cap skipped.
    (b) detect_duplicates с check_window=True — scoring-путь, window cap active.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipe = _build_news_pipeline(Path(self.tmp))
        self.flt = self.pipe.llm_filter

    def tearDown(self):
        _unmock()
        self.pipe.storage.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_digest_cluster_path_skips_window_cap(self):
        """
        (a) _cluster_deferred_duplicates вызывает detect_duplicates(check_window=False).
        При window exceeded + daily OK — дедуп вызывается.
        """
        items = [
            _mk_item(url=f"https://e.com/{i}", title=f"Item{i}", llm_score=6)
            for i in range(3)
        ]
        # Мокаем detect_duplicates на входе (entry guard)
        self.flt.detect_duplicates = MagicMock(return_value=set())
        # Подменяем _check_daily_cost: check_window=False → True, check_window=True → False
        def fake_cost_check(check_window=True):
            return not check_window  # False→True, True→False
        with patch.object(self.flt, '_check_daily_cost', side_effect=fake_cost_check):
            # Вызов через pipeline._cluster_deferred_duplicates (digest path)
            result_dup, _ = self.pipe._cluster_deferred_duplicates(
                items, "2026-01-01T00:00:00+00:00"
            )
        # detect_duplicates вызван (window cap пропущен)
        self.flt.detect_duplicates.assert_called_once()
        call_kwargs = self.flt.detect_duplicates.call_args
        # Проверяем check_window=False в вызове
        self.assertFalse(call_kwargs[1].get("check_window", True),
            "Digest path должен передавать check_window=False")

    def test_scoring_path_uses_window_cap(self):
        """
        (b) Scoring-путь (_run_llm_dedup) вызывает detect_duplicates
        с дефолтным check_window=True.
        """
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        win_idx = datetime.now(timezone.utc).hour // 6

        # Устанавливаем лимиты и cost так чтобы window exceeded
        self.flt._daily_cost_limit = 0.50
        self.flt._window_cost_limit = 0.12
        meta_store = {
            f"llm_daily_cost_{today}": "0.10",   # OK
            f"llm_daily_cost_{today}_w{win_idx}": "0.20",  # > limit
        }
        self.flt._storage = MagicMock()
        self.flt._storage.get_meta = MagicMock(side_effect=lambda k: meta_store.get(k))
        self.flt._storage.count_by_state = MagicMock(return_value=0)
        self.flt._storage.set_meta = MagicMock()

        items = [
            _mk_item(url=f"https://e.com/{i}", title=f"Item{i}", llm_score=6)
            for i in range(3)
        ]

        # detect_duplicates с дефолтным check_window=True
        # При window exceeded → early return без LLM-вызова
        result = self.flt.detect_duplicates(items, [])
        self.assertEqual(result, set(),
            "Scoring-дедуп при window exceeded должен вернуть пустое множество")


# =============================================================================
# Test: Minor-2 — ukr/bel letters write-back (GLOB расширен)
# =============================================================================

class TestUkrBelLetterWriteBack(unittest.TestCase):
    """
    Minor-2-fix: GLOB '*[А-Яа-яЁёІіЇїЄєЎўҐґ]*' покрывает укр./бел. буквы.
    llm_title с 'і' попадает в перевод и write-back обновляет БД
    (без бесконечного повторного перевода).
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipe = _build_news_pipeline(Path(self.tmp))
        self.flt = self.pipe.llm_filter

    def tearDown(self):
        _unmock()
        self.pipe.storage.close()
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ukrainian_llm_title_write_back_updates(self):
        """
        llm_title='Шторм з дощем' → update_translate_batch перезаписывает
        в БД (GLOB покрывает 'і', 'є' и др.).
        Без расширения GLOB — EN-перевод не перезаписал бы русский llm_title.
        """
        item = _mk_item(
            url="https://e.com/ukr1",
            title="Шторм",
            llm_score=6,
            llm_title="Шторм з дощем",  # ukr буква 'і'
        )
        self.pipe.storage.save(item)

        # EN-перевод
        n = self.pipe.storage.update_translate_batch([{
            "url": "https://e.com/ukr1",
            "title_en": "Storm with Rain",
            "summary_en": "Summary.",
        }])
        self.assertEqual(n, 1, "Write-back должен обновить строку с ukr буквой")

        # Перечитываем — llm_title обновлён
        [reloaded] = self.pipe.storage.get_all_by_state("new")
        self.assertEqual(reloaded.llm_title, "Storm with Rain",
            "EN-перевод должен перезаписать ukr llm_title")

    def test_ukrainian_llm_title_no_infinite_retranslate(self):
        """
        Повторный update_translate_batch с другим EN title НЕ перезаписывает
        уже EN llm_title (GLOB не match'ит 'i'/'r' и т.д.).
        """
        item = _mk_item(
            url="https://e.com/ukr2",
            title="Шторм",
            llm_score=6,
            llm_title="Шторм з дощем",
        )
        self.pipe.storage.save(item)

        # Первый EN перевод
        self.pipe.storage.update_translate_batch([{
            "url": "https://e.com/ukr2",
            "title_en": "Storm with Rain",
            "summary_en": "",
        }])
        # Повторный с другим title — НЕ перезаписывает (already EN)
        n2 = self.pipe.storage.update_translate_batch([{
            "url": "https://e.com/ukr2",
            "title_en": "Different Title",
            "summary_en": "",
        }])
        self.assertEqual(n2, 0, "EN llm_title не должен перезаписываться")

        [reloaded] = self.pipe.storage.get_all_by_state("new")
        self.assertEqual(reloaded.llm_title, "Storm with Rain",
            "Оригинальный EN llm_title должен сохраниться")


if __name__ == "__main__":
    unittest.main()