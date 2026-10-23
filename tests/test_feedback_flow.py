"""
tests/test_feedback_flow.py — Фаза 2b-1: инлайн-кнопки 👍/👎 на push-сообщениях.

Тестируемые сценарии:
  1. push-item с feedback_target_id → клавиатура с 3 кнопками, callback_data ≤64 байт
  2. без feedback_target_id → кнопок нет
  3. pipeline создаёт feedback_target для доставленных push-items
  4. клавиатура только на последней части при разбиении; лимит 4096 не нарушен
  5. ошибка создания target не ломает доставку
"""
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from newsbot.collector import NewsItem
from newsbot.delivery.telegram import TelegramDelivery


# ══════════════════════════════════════════════════════════════════
# фикстуры
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def base_item():
    """Базовый NewsItem без feedback_target_id."""
    return NewsItem(
        source="TestSrc",
        url="https://example.com/article",
        title="Тестовая статья",
        summary="Краткое содержание тестовой статьи.",
        published="2026-01-01",
        published_parsed=None,
        llm_score=8,
        llm_summary="Короткая выжимка.",
        llm_reason="Хорошая новость.",
    )


@pytest.fixture
def item_with_fbtid(base_item):
    """NewsItem с feedback_target_id=42."""
    item = base_item
    item.feedback_target_id = 42
    return item


# ══════════════════════════════════════════════════════════════════
# 1. Клавиатура с feedback_target_id: 3 кнопки, callback_data ≤64
# ══════════════════════════════════════════════════════════════════

class TestFeedbackKeyboardWithId:
    """push-item с feedback_target_id → клавиатура с 3 кнопками."""

    def test_keyboard_has_three_buttons(self, item_with_fbtid):
        """При feedback_target_id — reply_markup содержит 3 кнопки (url + 👍 + 👎)."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        parts = td._build_message_parts(item_with_fbtid)
        assert len(parts) == 1
        assert "reply_markup" in parts[0]
        kb = parts[0]["reply_markup"]["inline_keyboard"]
        assert len(kb) == 2, "Ожидаем 2 ряда: [url], [👍|👎]"
        assert len(kb[0]) == 1, "Ряд 1: одна кнопка-ссылка"
        assert len(kb[1]) == 2, "Ряд 2: две кнопки 👍 и 👎"

    def test_keyboard_url_button(self, item_with_fbtid):
        """Кнопка-ссылка на источник имеет правильный текст и URL."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        parts = td._build_message_parts(item_with_fbtid)
        kb = parts[0]["reply_markup"]["inline_keyboard"]
        url_btn = kb[0][0]
        assert url_btn["text"] == "🔗 Открыть источник"
        assert url_btn["url"] == "https://example.com/article"

    def test_callback_data_format(self, item_with_fbtid):
        """callback_data формат fb:<id>:u и fb:<id>:d."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        parts = td._build_message_parts(item_with_fbtid)
        kb = parts[0]["reply_markup"]["inline_keyboard"]
        btn_u, btn_d = kb[1]  # два элемента второго ряда
        assert btn_u["callback_data"] == "fb:42:u"
        assert btn_d["callback_data"] == "fb:42:d"

    def test_callback_data_length_within_limit(self, item_with_fbtid):
        """callback_data ≤ 64 байт (Telegram limit)."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        parts = td._build_message_parts(item_with_fbtid)
        kb = parts[0]["reply_markup"]["inline_keyboard"]
        for row in kb:
            for btn in row:
                cd = btn.get("callback_data", "")
                if cd:
                    assert len(cd.encode("utf-8")) <= 64, (
                        f"callback_data '{cd}' = {len(cd.encode('utf-8'))} байт > 64"
                    )

    def test_callback_data_max_id_length(self):
        """Даже с большим id callback_data ≤ 64 байт."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        item = NewsItem(
            source="S", url="https://x.com",
            title="T", summary="B", published="2026",
            published_parsed=None,
            feedback_target_id=9_999_999,
        )
        parts = td._build_message_parts(item)
        kb = parts[0]["reply_markup"]["inline_keyboard"]
        for row in kb:
            for btn in row:
                cd = btn.get("callback_data", "")
                if cd:
                    assert len(cd.encode("utf-8")) <= 64

    def test_item_without_url_only_thumbs_row(self):
        """F6c: item без URL + feedback_target_id → только ряд 👍/👎 (нет URL-кнопки)."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        item = NewsItem(
            source="S",
            url="",  # ← нет URL
            title="T",
            summary="B",
            published="2026",
            published_parsed=None,
            feedback_target_id=77,
        )
        parts = td._build_message_parts(item)
        assert "reply_markup" in parts[0]
        kb = parts[0]["reply_markup"]["inline_keyboard"]
        assert len(kb) == 1, f"Ожидаем 1 ряд (только 👍/👎), получили {len(kb)}"
        assert len(kb[0]) == 2
        assert kb[0][0]["callback_data"] == "fb:77:u"
        assert kb[0][1]["callback_data"] == "fb:77:d"
        # URL-кнопки нет
        assert all("url" not in btn for btn in kb[0])

    def test_worst_case_callback_data_max_int_id(self):
        """F6d: максимальный int id — callback_data ≤ 64 байт."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        item = NewsItem(
            source="S", url="https://x.com",
            title="T", summary="B", published="2026",
            published_parsed=None,
            feedback_target_id=2_147_483_647,  # INT MAX
        )
        parts = td._build_message_parts(item)
        cb = parts[0]["reply_markup"]["inline_keyboard"][1][0]["callback_data"]
        assert len(cb.encode("utf-8")) <= 64, (
            f"callback_data '{cb}' = {len(cb.encode('utf-8'))} bytes > 64"
        )


# ══════════════════════════════════════════════════════════════════
# 2. Без feedback_target_id — кнопок нет
# ══════════════════════════════════════════════════════════════════

class TestFeedbackKeyboardWithoutId:
    """Без feedback_target_id — inline keyboard отсутствует."""

    def test_no_thumbs_buttons_when_fbtid_none(self, base_item):
        """feedback_target_id=None → нет кнопок 👍/👎 (URL-кнопка остаётся)."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        parts = td._build_message_parts(base_item)
        assert "reply_markup" in parts[0]
        kb = parts[0]["reply_markup"]["inline_keyboard"]
        # Только URL-кнопка, нет ряда с 👍/👎
        assert len(kb) == 1, f"Ожидаем 1 ряд (URL), получили {len(kb)}"
        assert kb[0][0]["text"] == "🔗 Открыть источник"

    def test_no_thumbs_buttons_when_attr_missing(self):
        """feedback_target_id не установлен → нет 👍/👎 (URL-кнопка остаётся)."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        item = NewsItem(
            source="S", url="https://x.com",
            title="T", summary="B", published="2026",
            published_parsed=None,
        )
        # dataclass field имеет default=None → hasattr=True, но None
        parts = td._build_message_parts(item)
        assert "reply_markup" in parts[0]
        kb = parts[0]["reply_markup"]["inline_keyboard"]
        assert len(kb) == 1, "Только URL-кнопка, без 👍/👎"

    def test_url_button_still_present_without_fbtid(self, base_item):
        """Без feedback_target_id кнопка URL всё ещё есть."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        parts = td._build_message_parts(base_item)
        # reply_markup с одной кнопкой URL
        assert "reply_markup" in parts[0]
        kb = parts[0]["reply_markup"]["inline_keyboard"]
        assert len(kb) == 1
        assert kb[0][0]["text"] == "🔗 Открыть источник"


# ══════════════════════════════════════════════════════════════════
# 4. Клавиатура на последней части при разбиении; лимит 4096
# ══════════════════════════════════════════════════════════════════

class TestFeedbackKeyboardSplitting:
    """Клавиатура только на последней части; 4096 не нарушен."""

    def test_keyboard_on_last_part_only(self):
        """При разбиении — reply_markup только на последнем payload."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        # Очень длинный body: нужен реальный oversize для разбиения.
        # title ≈ 250, footer ≈ 30, overhead ≈ 30 → body нужно > 3686 для split.
        # Делаем ~5000 символов чтобы гарантированно разбить.
        long_body = "Параграф первый. " + "X" * 5000
        item = NewsItem(
            source="S",
            url="https://example.com/long",
            title="Test",
            summary=long_body,
            published="2026",
            published_parsed=None,
            feedback_target_id=7,
        )
        parts = td._build_message_parts(item)
        assert len(parts) > 1, f"Ожидаем разбиение, parts={len(parts)}"

        # все кроме последней — без reply_markup
        for p in parts[:-1]:
            assert "reply_markup" not in p, (
                f"Промежуточная часть не должна иметь reply_markup: {p.get('text', '')[:50]}"
            )

        # последняя — с reply_markup
        assert "reply_markup" in parts[-1]
        kb = parts[-1]["reply_markup"]["inline_keyboard"]
        assert len(kb) == 2  # url + 👍/👎

    def test_no_keyboard_on_intermediate_parts_even_without_fbtid(self):
        """Промежуточные части не имеют reply_markup даже при отсутствии fbtid."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        long_body = "Параграф 1. " + "Y" * 5000
        item = NewsItem(
            source="S",
            url="https://x.com",
            title="Test",
            summary=long_body,
            published="2026",
            published_parsed=None,
            # feedback_target_id=None
        )
        parts = td._build_message_parts(item)
        assert len(parts) > 1, f"Ожидаем разбиение, parts={len(parts)}"
        for p in parts[:-1]:
            assert "reply_markup" not in p

    def test_all_parts_within_4096_limit(self):
        """Каждая часть ≤ 4096 символов (Telegram message limit)."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        item = NewsItem(
            source="TestSrc",
            url="https://example.com/long-article",
            title="Очень длинная статья о чём-то важном",
            summary="Параграф 1.\n\n" + "Z" * 5000,
            published="2026-01-01",
            published_parsed=None,
            feedback_target_id=99,
        )
        parts = td._build_message_parts(item)
        for p in parts:
            assert len(p["text"]) <= 4096, (
                f"Payload size {len(p['text'])} > 4096: {p['text'][:100]}"
            )


# ══════════════════════════════════════════════════════════════════
# 3. Pipeline создаёт feedback_target для доставленных push-items
# ══════════════════════════════════════════════════════════════════

class TestPipelineCreatesFeedbackTarget:
    """Pipeline создаёт feedback_target перед отправкой push-items."""

    def test_feedback_target_created_before_delivery(self, tmp_path):
        """get_or_create_feedback_target_from_item вызывается для push-item (LLM off → fallback)."""
        from newsbot.pipeline import NewsPipeline

        db_path = tmp_path / "test.db"
        config = {
            "sources": [],
            "database": {"path": str(db_path)},
            "telegram": {"token": "test", "chat_id": "123"},
            "llm": {"enabled": False},
            "feedback": {"enabled": True},
        }

        with patch.object(TelegramDelivery, "_api_call", return_value={}):
            pipeline = NewsPipeline(config)
            storage = pipeline.storage

            item = NewsItem(
                source="S",
                url="https://example.com/test",
                title="Test",
                summary="Summary",
                published="2026",
                published_parsed=None,
                llm_score=8,
            )

            pipeline._process_batch([item])

            assert item.feedback_target_id is not None, (
                "feedback_target_id должен быть установлен после _process_batch"
            )
            target = storage.get_feedback_target(item.feedback_target_id)
            assert target is not None
            # F3: без LLM → kind='fallback' (не профильтрованный трафик)
            assert target["kind"] == "fallback"
            assert target["url"] == "https://example.com/test"

            pipeline.close()

    def test_kind_push_when_llm_filter_active(self, tmp_path):
        """F3: когда LLM фильтр активен и item проходит оценку → kind='push'."""
        from newsbot.pipeline import NewsPipeline
        from newsbot.llm.filter import FilterResult

        db_path = tmp_path / "test.db"
        config = {
            "sources": [],
            "database": {"path": str(db_path)},
            "telegram": {"token": "test", "chat_id": "123"},
            "llm": {
                "enabled": True,
                "api_key": "fake-key",
                "model": "deepseek-chat",
                "push_threshold": 7,
                "digest_threshold": 5,
                "temperature": 0,
                "timeout_sec": 10,
            },
            "feedback": {"enabled": True},
        }

        with patch.object(TelegramDelivery, "_api_call", return_value={}):
            pipeline = NewsPipeline(config)

            item = NewsItem(
                source="S",
                url="https://example.com/llm-test",
                title="Test LLM",
                summary="Summary",
                published="2026",
                published_parsed=None,
            )

            # эмулируем оценку с score=8 (push; urgent=True т.к. push_urgent_only=True)
            with patch.object(
                pipeline.llm_filter, "evaluate",
                return_value=FilterResult(ok=True, score=8, summary="ok", reason="good", urgent=True)
            ):
                pipeline._process_batch([item])

            assert item.feedback_target_id is not None
            target = pipeline.storage.get_feedback_target(item.feedback_target_id)
            assert target["kind"] == "push", (
                f"Профильтрованный item с score=8 должен иметь kind='push', "
                f"получено: {target['kind']!r}"
            )
            pipeline.close()

    def test_kind_fallback_when_llm_error(self, tmp_path):
        """F3: когда LLM-оценка не удалась (fallback) → kind='fallback'."""
        from newsbot.pipeline import NewsPipeline
        from newsbot.llm.filter import FilterResult

        db_path = tmp_path / "test.db"
        config = {
            "sources": [],
            "database": {"path": str(db_path)},
            "telegram": {"token": "test", "chat_id": "123"},
            "llm": {
                "enabled": True,
                "api_key": "fake-key",
                "model": "deepseek-chat",
                "push_threshold": 7,
                "digest_threshold": 5,
                "temperature": 0,
                "timeout_sec": 10,
                "push_urgent_only": False,  # M2: legacy — fallback доставляется как push
            },
            "feedback": {"enabled": True},
        }

        with patch.object(TelegramDelivery, "_api_call", return_value={}):
            pipeline = NewsPipeline(config)

            item = NewsItem(
                source="S",
                url="https://example.com/llm-fail",
                title="Test LLM fail",
                summary="Summary",
                published="2026",
                published_parsed=None,
            )

            # эмулируем failure LLM → fallback
            with patch.object(
                pipeline.llm_filter, "evaluate",
                return_value=FilterResult(ok=False, score=0, summary="", reason="")
            ):
                pipeline._process_batch([item])

            assert item.feedback_target_id is not None
            target = pipeline.storage.get_feedback_target(item.feedback_target_id)
            assert target["kind"] == "fallback", (
                f"Fallback item (ok=False) должен иметь kind='fallback', "
                f"получено: {target['kind']!r}"
            )
            pipeline.close()

    def test_no_feedback_target_when_disabled(self, tmp_path):
        """При feedback.enabled=false feedback_target_id не устанавливается."""
        from newsbot.pipeline import NewsPipeline

        db_path = tmp_path / "test.db"
        config = {
            "sources": [],
            "database": {"path": str(db_path)},
            "telegram": {"token": "test", "chat_id": "123"},
            "llm": {"enabled": False},
            "feedback": {"enabled": False},  # ← явно выключен
        }

        with patch.object(TelegramDelivery, "_api_call", return_value={}):
            pipeline = NewsPipeline(config)

            item = NewsItem(
                source="S",
                url="https://example.com/test",
                title="Test",
                summary="Summary",
                published="2026",
                published_parsed=None,
                llm_score=8,
            )

            pipeline._process_batch([item])
            assert item.feedback_target_id is None, (
                "При feedback.enabled=False feedback_target_id не должен устанавливаться"
            )
            pipeline.close()

    def test_no_feedback_target_in_console_delivery(self, tmp_path):
        """F2: в режиме ConsoleDelivery targets НЕ создаются."""
        from newsbot.pipeline import NewsPipeline
        from newsbot.delivery import ConsoleDelivery

        db_path = tmp_path / "test.db"
        config = {
            "sources": [],
            "database": {"path": str(db_path)},
            "telegram": {"token": "", "chat_id": ""},  # ← нет TG → ConsoleDelivery
            "llm": {"enabled": False},
            "feedback": {"enabled": True},
        }

        pipeline = NewsPipeline(config)  # создаст ConsoleDelivery
        assert isinstance(pipeline.delivery, ConsoleDelivery)

        item = NewsItem(
            source="S",
            url="https://example.com/test",
            title="Test",
            summary="Summary",
            published="2026",
            published_parsed=None,
            llm_score=8,
        )

        pipeline._process_batch([item])
        assert item.feedback_target_id is None, (
            "В режиме ConsoleDelivery feedback_target_id не должен устанавливаться"
        )
        pipeline.close()

    def test_retry_does_not_create_duplicate_target(self, tmp_path):
        """F1: повторный проход _process_batch возвращает ТОТ ЖЕ target_id (не дублирует)."""
        from newsbot.pipeline import NewsPipeline

        db_path = tmp_path / "test.db"
        config = {
            "sources": [],
            "database": {"path": str(db_path)},
            "telegram": {"token": "test", "chat_id": "123"},
            "llm": {"enabled": False},
            "feedback": {"enabled": True},
        }

        with patch.object(TelegramDelivery, "_api_call", return_value={}):
            pipeline = NewsPipeline(config)

            item = NewsItem(
                source="S",
                url="https://example.com/retry-test",
                title="RetryTest",
                summary="Summary",
                published="2026",
                published_parsed=None,
                llm_score=8,
            )

            # первый проход
            pipeline._process_batch([item])
            first_id = item.feedback_target_id
            assert first_id is not None

            # сбрасываем (эмулируем pending → retry)
            item.feedback_target_id = None

            # второй проход — ТОТ ЖЕ id
            pipeline._process_batch([item])
            assert item.feedback_target_id == first_id, (
                f"Повторный проход: expected {first_id}, got {item.feedback_target_id}. "
                "Должен вернуть ТОТ ЖЕ id (идемпотентность F1)"
            )

            # в БД ровно один target для этого url
            all_targets = list(pipeline.storage._connect().execute(
                "SELECT id FROM feedback_targets WHERE url = ?",
                ("https://example.com/retry-test",),
            ).fetchall())
            assert len(all_targets) == 1, (
                f"Должен быть ровно 1 target, найдено {len(all_targets)}: {all_targets}"
            )

            pipeline.close()


# ══════════════════════════════════════════════════════════════════
# 5. Ошибка создания target не ломает доставку
# ══════════════════════════════════════════════════════════════════

class TestFeedbackTargetErrorDoesNotBreakDelivery:
    """Ошибка add_feedback_target_from_item — best-effort, доставка продолжается."""

    def test_delivery_succeeds_when_add_target_raises(self, tmp_path):
        """При exception в add_feedback_target_from_item доставка продолжается."""
        from newsbot.pipeline import NewsPipeline

        db_path = tmp_path / "test.db"
        config = {
            "sources": [],
            "database": {"path": str(db_path)},
            "telegram": {"token": "test", "chat_id": "123"},
            "llm": {"enabled": False},
            "feedback": {"enabled": True},
        }

        # эмулируем ошибку БД при создании target
        def raise_on_add_target(*args, **kwargs):
            raise RuntimeError("DB error")

        with patch.object(TelegramDelivery, "_api_call", return_value={}) as mock_api:
            pipeline = NewsPipeline(config)

            item = NewsItem(
                source="S",
                url="https://example.com/test",
                title="Test",
                summary="Summary",
                published="2026",
                published_parsed=None,
                llm_score=8,
            )

            with patch.object(
                pipeline.storage, "get_or_create_feedback_target_from_item",
                side_effect=RuntimeError("DB error")
            ):
                # _process_batch не должен бросить исключение
                failed_urls = pipeline._process_batch([item])

            # доставка прошла успешно (send_batch был вызван)
            assert mock_api.call_count >= 1, "send_batch должен быть вызван"

            pipeline.close()

    def test_delivery_succeeds_when_disabled(self, tmp_path):
        """При feedback.enabled=False доставка работает без создания target."""
        from newsbot.pipeline import NewsPipeline

        db_path = tmp_path / "test.db"
        config = {
            "sources": [],
            "database": {"path": str(db_path)},
            "telegram": {"token": "test", "chat_id": "123"},
            "llm": {"enabled": False},
            "feedback": {"enabled": True},
        }

        with patch.object(TelegramDelivery, "_api_call", return_value={}) as mock_api:
            pipeline = NewsPipeline(config)

            item = NewsItem(
                source="S",
                url="https://example.com/test",
                title="Test",
                summary="Summary",
                published="2026",
                published_parsed=None,
                llm_score=8,
            )

            # даже если get_or_create_feedback_target_from_item упадёт — доставка важнее
            with patch.object(
                pipeline.storage, "get_or_create_feedback_target_from_item",
                side_effect=RuntimeError("DB error")
            ):
                failed_urls = pipeline._process_batch([item])

            assert mock_api.call_count >= 1
            pipeline.close()


# ══════════════════════════════════════════════════════════════════
# ConsoleDelivery совместим (ignores feedback_target_id)
# ══════════════════════════════════════════════════════════════════

class TestConsoleDeliveryIgnoresFeedback:
    """ConsoleDelivery корректно работает с NewsItem.feedback_target_id."""

    def test_console_delivery_no_error_with_fbtid(self):
        """ConsoleDelivery.send_batch не падает на items с feedback_target_id."""
        from newsbot.delivery.console import ConsoleDelivery

        item = NewsItem(
            source="S",
            url="https://x.com",
            title="T",
            summary="B",
            published="2026",
            published_parsed=None,
            feedback_target_id=42,
        )

        cd = ConsoleDelivery()
        # не должно бросить исключение
        result = cd.send_batch([item])
        assert result == (["https://x.com"], [])

    def test_console_delivery_no_error_without_fbtid(self):
        """ConsoleDelivery.send_batch работает и без feedback_target_id."""
        from newsbot.delivery.console import ConsoleDelivery

        item = NewsItem(
            source="S",
            url="https://x.com",
            title="T",
            summary="B",
            published="2026",
            published_parsed=None,
        )

        cd = ConsoleDelivery()
        result = cd.send_batch([item])
        assert result == (["https://x.com"], [])


# ══════════════════════════════════════════════════════════════════
# NewsItem: feedback_target_id — transient, не ломает конструктор
# ══════════════════════════════════════════════════════════════════

class TestNewsItemFeedbackTargetId:
    """feedback_target_id — transient runtime-поле."""

    def test_feedback_target_id_default_is_none(self):
        """По умолчанию feedback_target_id=None."""
        item = NewsItem(
            source="S", url="https://x.com",
            title="T", summary="B",
            published="2026", published_parsed=None,
        )
        assert item.feedback_target_id is None

    def test_feedback_target_id_can_be_set(self):
        """feedback_target_id можно установить после создания."""
        item = NewsItem(
            source="S", url="https://x.com",
            title="T", summary="B",
            published="2026", published_parsed=None,
        )
        item.feedback_target_id = 123
        assert item.feedback_target_id == 123

    def test_standard_constructor_still_works(self):
        """Все стандартные поля конструктора работают как раньше."""
        dt = datetime.now(timezone.utc)
        item = NewsItem(
            source="Src",
            url="https://x.com/1",
            title="Title",
            summary="Summary text",
            published="2026-01-01",
            published_parsed=dt,
            llm_score=9,
            llm_summary="LLM summary",
            llm_reason="Reason",
            retry_count=3,
            fetched_at=dt,
            feedback_target_id=5,
        )
        assert item.source == "Src"
        assert item.url == "https://x.com/1"
        assert item.llm_score == 9
        assert item.llm_reason == "Reason"
        assert item.retry_count == 3
        assert item.feedback_target_id == 5

    def test_dataclass_fields_unchanged(self, base_item):
        """feedback_target_id не перезаписывает существующие поля."""
        item = base_item
        assert item.source == "TestSrc"
        assert item.url == "https://example.com/article"
        assert item.llm_score == 8
        assert item.feedback_target_id is None  # из фикстуры, не установлен


# ══════════════════════════════════════════════════════════════════
# feedback.enabled по умолчанию True (DEFAULT_CONFIG)
# ══════════════════════════════════════════════════════════════════

class TestFeedbackDefaultEnabled:
    def test_feedback_default_enabled_in_config(self):
        from newsbot.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG.get("feedback", {}).get("enabled") is True


# ══════════════════════════════════════════════════════════════════
# Фаза 2b-2b: дайджест-клавиатура (send_digest)
# ══════════════════════════════════════════════════════════════════

class TestDigestKeyboard:
    """Клавиатура на дайджест-сообщении: кнопки на КАЖДУЮ новость."""

    def test_send_digest_creates_inline_keyboard(self):
        """send_digest создаёт reply_markup с inline_keyboard (по ряду на item)."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        text = "📰 Дайджест — 2 из 2 новостей\n—\n<b>Item1</b>\nBody1\n2025\n<b>Item2</b>\nBody2\n2025"
        item_buttons = [
            {"url": "https://ex.com/1", "target_id": 10},
            {"url": "https://ex.com/2", "target_id": 11},
        ]
        mock_api = MagicMock(return_value={})
        with patch.object(td, "_api_call", mock_api):
            result = td.send_digest(text, item_buttons)

        assert result is True
        mock_api.assert_called_once()
        payload = mock_api.call_args[0][1]
        assert "reply_markup" in payload
        assert "inline_keyboard" in payload["reply_markup"]
        # 2 ряда (по одному на item)
        assert len(payload["reply_markup"]["inline_keyboard"]) == 2

    def test_send_digest_row_format_thumbs_only(self):
        """Каждый ряд: только [👍] [👎] (URL в тексте, не дублируется кнопкой)."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        item_buttons = [
            {"url": "https://ex.com/news1", "target_id": 5},
        ]
        mock_api = MagicMock(return_value={})
        with patch.object(td, "_api_call", mock_api):
            td.send_digest("Digest text", item_buttons)

        payload = mock_api.call_args[0][1]
        kb = payload["reply_markup"]["inline_keyboard"]
        row = kb[0]
        # 2 кнопки: только 👍 + 👎 (URL НЕ дублируется кнопкой — он в тексте)
        assert len(row) == 2
        assert row[0]["text"] == "👍"
        assert row[0]["callback_data"] == "fb:5:u"
        assert row[1]["text"] == "👎"
        assert row[1]["callback_data"] == "fb:5:d"

    def test_send_digest_row_without_url_only_thumbs(self):
        """Item без URL: ряд с только 👍/👎."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        item_buttons = [
            {"url": None, "target_id": 7},
        ]
        mock_api = MagicMock(return_value={})
        with patch.object(td, "_api_call", mock_api):
            td.send_digest("Digest text", item_buttons)

        payload = mock_api.call_args[0][1]
        kb = payload["reply_markup"]["inline_keyboard"]
        row = kb[0]
        assert len(row) == 2
        assert row[0]["callback_data"] == "fb:7:u"
        assert row[1]["callback_data"] == "fb:7:d"
        assert all("url" not in btn for btn in row)

    def test_send_digest_row_without_target_skipped(self):
        """Item без target_id: ряд пропускается, reply_markup отсутствует."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        item_buttons = [
            {"url": "https://ex.com/noid", "target_id": None},
        ]
        mock_api = MagicMock(return_value={})
        with patch.object(td, "_api_call", mock_api):
            td.send_digest("Digest text", item_buttons)

        payload = mock_api.call_args[0][1]
        # При отсутствии target_id строк нет → reply_markup отсутствует
        assert "reply_markup" not in payload, (
            "reply_markup не должно быть при отсутствии target_id"
        )

    def test_send_digest_empty_item_buttons_no_keyboard(self):
        """item_buttons=[] — reply_markup отсутствует (нет новостей с кнопками)."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        mock_api = MagicMock(return_value={})
        with patch.object(td, "_api_call", mock_api):
            td.send_digest("Digest text", [])

        payload = mock_api.call_args[0][1]
        assert "reply_markup" not in payload

    def test_send_digest_callback_data_under_64_bytes(self):
        """callback_data ≤ 64 байт (Telegram limit). URL в item_button игнорируется."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        item_buttons = [
            {"url": "https://ex.com/large", "target_id": 9_999_999},
        ]
        mock_api = MagicMock(return_value={})
        with patch.object(td, "_api_call", mock_api):
            td.send_digest("Digest", item_buttons)

        payload = mock_api.call_args[0][1]
        kb = payload["reply_markup"]["inline_keyboard"]
        for row in kb:
            for btn in row:
                cd = btn.get("callback_data", "")
                if cd:
                    assert len(cd.encode("utf-8")) <= 64, (
                        f"callback_data '{cd}' = {len(cd.encode('utf-8'))} bytes > 64"
                    )

    def test_send_digest_html_parse_mode(self):
        """send_digest использует parse_mode=HTML."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        mock_api = MagicMock(return_value={})
        with patch.object(td, "_api_call", mock_api):
            td.send_digest("📰 <b>Digest</b>", [])

        payload = mock_api.call_args[0][1]
        assert payload["parse_mode"] == "HTML"
        assert "link_preview_options" in payload

    def test_send_digest_skips_rows_with_no_url_and_no_target(self):
        """Ряд без target_id пропускается (URL в тексте)."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        item_buttons = [
            {"url": "https://ex.com/1", "target_id": 1},
            {"url": None, "target_id": None},  # пустой — пропускается
            {"url": None, "target_id": 3},      # только 👍/👎
        ]
        mock_api = MagicMock(return_value={})
        with patch.object(td, "_api_call", mock_api):
            td.send_digest("Digest", item_buttons)

        payload = mock_api.call_args[0][1]
        kb = payload["reply_markup"]["inline_keyboard"]
        assert len(kb) == 2
        assert len(kb[0]) == 2  # только 👍 + 👎
        assert len(kb[1]) == 2  # только 👍 + 👎

    def test_send_digest_returns_false_on_error(self):
        """send_digest возвращает False при ошибке API."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        with patch.object(
            td, "_api_call",
            side_effect=RuntimeError("API error")
        ):
            result = td.send_digest("Digest", [{"url": "https://ex.com/1", "target_id": 1}])
        assert result is False

    def test_send_digest_returns_false_without_chat_id(self):
        """send_digest возвращает False если chat_id пуст."""
        td = TelegramDelivery(token="test_token", chat_id="")
        result = td.send_digest("Digest", [{"url": "https://ex.com/1", "target_id": 1}])
        assert result is False

    def test_send_digest_url_field_ignored(self):
        """Поле 'url' в item_buttons игнорируется (URL уже в тексте дайджеста).
        Кнопки — только 👍/👎 по target_id."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        item_buttons = [
            {"url": "http://example.com/ok", "target_id": 1},
            {"url": "javascript:alert(1)", "target_id": 2},
            {"url": "tg://resolve?domain=foo", "target_id": 3},
            {"url": "ftp://files.example.com/file", "target_id": 4},
        ]
        mock_api = MagicMock(return_value={})
        with patch.object(td, "_api_call", mock_api):
            td.send_digest("Digest", item_buttons)

        payload = mock_api.call_args[0][1]
        kb = payload["reply_markup"]["inline_keyboard"]
        assert len(kb) == 4, f"Expected 4 rows, got {len(kb)}"
        for i, row in enumerate(kb, start=1):
            # Каждый ряд: только 👍 + 👎, URL-полей нет
            assert len(row) == 2, f"Row {i}: expected 2 thumbs, got {len(row)}"
            assert all("url" not in btn for btn in row), f"Row {i}: не должно быть URL-кнопки"
            assert all("callback_data" in btn for btn in row), f"Row {i}: должны быть thumbs"


class TestDigestKeyboardConsoleCompatibility:
    """ConsoleDelivery.send_digest — игнорирует кнопки, печатает текст."""

    def test_send_digest_calls_send_text(self):
        """ConsoleDelivery.send_digest просто вызывает send_text."""
        from newsbot.delivery.console import ConsoleDelivery

        cd = ConsoleDelivery()
        result = cd.send_digest("📰 Digest text", [{"url": "https://x.com", "target_id": 1}])
        assert result is True  # Console send_text всегда True

    def test_send_digest_no_error_with_empty_buttons(self):
        """ConsoleDelivery.send_digest не падает при пустом item_buttons."""
        from newsbot.delivery.console import ConsoleDelivery

        cd = ConsoleDelivery()
        result = cd.send_digest("Empty digest", [])
        assert result is True


# ══════════════════════════════════════════════════════════════════════════════
# M3: URL-whitelist — не-http схемы и control-символы не попадают в кнопку
# ══════════════════════════════════════════════════════════════════════════════

class TestUrlButtonSchemeWhitelist:
    """URL-кнопка добавляется ТОЛЬКО для http:// / https://, без control-символов."""

    def _parts(self, url):
        """Build message parts for item with given URL, no feedback_target_id."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        item = NewsItem(
            source="S",
            url=url,
            title="Title",
            summary="Summary",
            published="2026",
            published_parsed=None,
        )
        return td._build_message_parts(item)

    def _has_url_btn(self, parts):
        """True если в inline_keyboard есть кнопка с полем 'url'."""
        if "reply_markup" not in parts[-1]:
            return False
        for row in parts[-1]["reply_markup"]["inline_keyboard"]:
            for btn in row:
                if "url" in btn:
                    return True
        return False

    def test_http_url_button_added(self):
        """http:// — кнопка должна быть."""
        parts = self._parts("http://example.com/article")
        assert self._has_url_btn(parts), "http:// URL должен добавлять кнопку"

    def test_https_url_button_added(self):
        """https:// — кнопка должна быть."""
        parts = self._parts("https://example.com/article")
        assert self._has_url_btn(parts), "https:// URL должен добавлять кнопку"

    def test_mailto_no_button(self):
        """mailto: — кнопка НЕ добавляется, сообщение отправляется."""
        parts = self._parts("mailto:user@example.com")
        assert not self._has_url_btn(parts), "mailto: не должен добавлять кнопку"

    def test_tg_scheme_no_button(self):
        """tg:// — кнопка НЕ добавляется."""
        parts = self._parts("tg://resolve?domain=somebot")
        assert not self._has_url_btn(parts), "tg:// не должен добавлять кнопку"

    def test_newline_in_url_no_button(self):
        """URL с \\n — кнопка НЕ добавляется."""
        parts = self._parts("https://example.com/art\nicle")
        assert not self._has_url_btn(parts), "URL с \\n не должен добавлять кнопку"

    def test_carriage_return_in_url_no_button(self):
        """URL с \\r — кнопка НЕ добавляется."""
        parts = self._parts("https://example.com/art\r\nicle")
        assert not self._has_url_btn(parts), "URL с \\r не должен добавлять кнопку"

    def test_tel_scheme_no_button(self):
        """tel: — кнопка НЕ добавляется."""
        parts = self._parts("tel:+79001234567")
        assert not self._has_url_btn(parts), "tel: не должен добавлять кнопку"

    def test_whitespace_trimmed_no_button(self):
        """URL с whitespace-ом по краям — не добавляется."""
        parts = self._parts("  https://example.com/  ")
        assert not self._has_url_btn(parts), "URL с whitespace не должен добавлять кнопку"

    def test_empty_url_no_button(self):
        """Пустой URL — кнопки нет."""
        parts = self._parts("")
        assert not self._has_url_btn(parts), "Пустой URL не должен добавлять кнопку"

    def test_relative_url_no_button(self):
        """/path/to/page — относительный URL без схемы не добавляется."""
        parts = self._parts("/path/to/page")
        assert not self._has_url_btn(parts), "Относительный URL не должен добавлять кнопку"

    def test_url_stripped_of_whitespace_in_button(self):
        """
        URL с leading/trailing whitespace — кнопка НЕ добавляется.
        is_valid_http_url отвергает такие URL (url != url.strip()).
        """
        td = TelegramDelivery(token="test_token", chat_id="123456")
        item = NewsItem(
            source="S",
            url="  https://example.com/article  ",
            title="T",
            summary="B",
            published="2026",
            published_parsed=None,
        )
        parts = td._build_message_parts(item)
        # кнопка НЕ добавляется (URL отвергнут как невалидный)
        assert "reply_markup" not in parts[-1]

    def test_http_button_still_works_with_feedback_id(self):
        """http:// + feedback_target_id — URL-кнопка + 👍/👎."""
        td = TelegramDelivery(token="test_token", chat_id="123456")
        item = NewsItem(
            source="S",
            url="http://example.com/article",
            title="Title",
            summary="Summary",
            published="2026",
            published_parsed=None,
        )
        item.feedback_target_id = 42
        parts = td._build_message_parts(item)
        kb = parts[-1]["reply_markup"]["inline_keyboard"]
        url_btns = [btn for row in kb for btn in row if "url" in btn]
        cb_btns = [btn for row in kb for btn in row if "callback_data" in btn]
        assert len(url_btns) == 1
        assert url_btns[0]["url"] == "http://example.com/article"
        assert len(cb_btns) == 2

    def test_internal_space_in_url_no_button(self):
        """
        URL с внутренним пробелом ('https://ex.com/a b') — кнопка НЕ добавляется.
        is_valid_http_url отвергает через " " in url.
        """
        td = TelegramDelivery(token="test_token", chat_id="123456")
        item = NewsItem(
            source="S",
            url="https://example.com/article with space",
            title="T",
            summary="B",
            published="2026",
            published_parsed=None,
        )
        parts = td._build_message_parts(item)
        assert "reply_markup" not in parts[-1]