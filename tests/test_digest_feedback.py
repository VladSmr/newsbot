"""
Тесты: отключение фидбэка в дайджесте (digest_enabled).

Покрывает:
  - digest_enabled=false (дефолт): _maybe_run_digest → send_text,
    feedback_targets НЕ создаются, send_digest НЕ вызывается
  - digest_enabled=true + feedback.enabled=true + TelegramDelivery:
    send_digest вызывается, кнопки — только 👍/👎 (без URL)
  - пуш-путь: send_batch по-прежнему добавляет 👍/👎
  - конфиг: feedback.digest_enabled парсится как bool, дефолт false

Запуск: python -m pytest tests/test_digest_feedback.py -v -p no:randomly
"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from newsbot.storage import Storage, NewsItem
from newsbot.pipeline import NewsPipeline
from newsbot.delivery.telegram import TelegramDelivery
from newsbot.digest import run_digest


# =============================================================================
# helpers
# =============================================================================

def _mk_item(source="Test", title="Test", url=None, score=6, **kw):
    if url is None:
        url = f"https://example.com/{hash(title)}"
    return NewsItem(
        source=source, url=url, title=title, summary="Summary.",
        published="2025-01-01", published_parsed=None,
        llm_score=score, llm_summary="Summary.", llm_reason="",
        fetched_at=datetime(2025, 1, 1, 12, 0, 0), **kw,
    )


def _tmp_storage():
    fd, path = tempfile.mkstemp(suffix=".db")
    import os
    os.close(fd)
    st = Storage(path)
    yield st
    st.close()
    Path(path).unlink(missing_ok=True)


# =============================================================================
# fixture: pipeline с мок-доставкой
# =============================================================================

class _MockTelegramDelivery(TelegramDelivery):
    """TelegramDelivery с моком HTTP — можно проверять вызовы."""

    def __init__(self):
        # настоящий __init__ не нужен (токен/chat_id не используются в моке)
        self.token = "mock_token"
        self.chat_id = "mock_chat"
        self._calls: list = []

    def _api_call(self, method, payload):
        self._calls.append((method, dict(payload)))
        return {"ok": True, "result": {"message_id": 1}}

    def reset_calls(self):
        self._calls.clear()


# =============================================================================
# Тест 1: digest_enabled=false — send_digest НЕ вызывается, send_text вместо
# =============================================================================

def test_digest_disabled_no_send_digest(tmp_path, monkeypatch):
    """При digest_enabled=false (дефолт) дайджест отправляется через send_text,
    send_digest НЕ вызывается, feedback_targets НЕ создаются."""
    db_path = str(tmp_path / "test.db")
    mock_delivery = _MockTelegramDelivery()

    cfg = {
        "sources": [],
        "database": {"path": db_path},
        "telegram": {"token": "x", "chat_id": "x"},
        "llm": {"enabled": False},
        "digest": {"enabled": True},
        "feedback": {
            "enabled": True,
            # digest_enabled отсутствует → дефолт false
        },
    }

    # Storage мок (не нужен реальный)
    mock_storage = MagicMock(spec=Storage)
    # get_deferred_items возвращает items
    item1 = _mk_item(title="News 1")
    item2 = _mk_item(title="News 2")
    mock_storage.get_deferred_items.return_value = [item1, item2]
    mock_storage.get_meta.return_value = None
    # get_or_create_feedback_target_from_item — счётчик вызовов
    mock_storage.get_or_create_feedback_target_from_item = MagicMock(return_value=1)

    from newsbot.pipeline import NewsPipeline
    # Подменяем delivery
    with monkeypatch.context() as m:
        # NewsPipeline создаёт delivery сам из конфига
        # Нам нужен мок-экземпляр
        pass

    # Создаём pipeline с моком storage
    class PatchedPipeline(NewsPipeline):
        def __init__(self, config, storage, delivery):
            # пропускаем __init__NewsPipeline, ставим напрямую
            self.config = config
            self.storage = storage
            self.delivery = delivery
            self.sources = config.get("sources", []) or []
            self.llm_filter = None
            self._feedback_context = None
            self._llm_dedup_skipped_urls = set()
            self._max_retries = 3
            self._llm_calls_left = 150
            self._max_per_cycle = 150
            self._off_peak_only = False

    pipeline = PatchedPipeline(cfg, mock_storage, mock_delivery)

    # Запускаем дайджест
    count = pipeline.run_digest(force=True)

    # send_digest НЕ вызывался
    send_digest_calls = [c for c in mock_delivery._calls if c[0] == "sendMessage"]
    # При send_text reply_markup отсутствует
    reply_markups = [c[1].get("reply_markup") for c in send_digest_calls]
    # send_text не имеет reply_markup — значит это send_text, не send_digest с кнопками
    assert all(r is None for r in reply_markups), (
        f"Ожидался send_text (без reply_markup), но есть reply_markup: {reply_markups}"
    )
    # feedback_targets НЕ создавались
    assert mock_storage.get_or_create_feedback_target_from_item.call_count == 0, (
        f"get_or_create_feedback_target_from_item вызван "
        f"{mock_storage.get_or_create_feedback_target_from_item.call_count} раз, "
        f"ожидалось 0"
    )
    # Хотя бы один sendMessage был
    assert len(send_digest_calls) > 0, "Ожидался хотя бы один вызов sendMessage (send_text)"


# =============================================================================
# Тест 2: digest_enabled=true → send_digest вызывается, кнопки — только 👍/👎
# =============================================================================

def test_digest_enabled_true_thumbs_only(tmp_path, monkeypatch):
    """При digest_enabled=true + TelegramDelivery: send_digest с кнопками
    только 👍/👎 (без URL)."""
    db_path = str(tmp_path / "test.db")
    mock_delivery = _MockTelegramDelivery()

    cfg = {
        "sources": [],
        "database": {"path": db_path},
        "telegram": {"token": "x", "chat_id": "x"},
        "llm": {"enabled": False},
        "digest": {"enabled": True},
        "feedback": {
            "enabled": True,
            "digest_enabled": True,  # ← включено
        },
    }

    mock_storage = MagicMock(spec=Storage)
    item1 = _mk_item(title="News A", url="https://example.com/a")
    item2 = _mk_item(title="News B", url="https://example.com/b")
    mock_storage.get_deferred_items.return_value = [item1, item2]
    mock_storage.get_meta.return_value = None
    mock_storage.get_or_create_feedback_target_from_item = MagicMock(
        side_effect=[10, 11]
    )

    class PatchedPipeline(NewsPipeline):
        def __init__(self, config, storage, delivery):
            self.config = config
            self.storage = storage
            self.delivery = delivery
            self.sources = config.get("sources", []) or []
            self.llm_filter = None
            self._feedback_context = None
            self._llm_dedup_skipped_urls = set()
            self._max_retries = 3
            self._llm_calls_left = 150
            self._max_per_cycle = 150
            self._off_peak_only = False

    pipeline = PatchedPipeline(cfg, mock_storage, mock_delivery)
    pipeline.run_digest(force=True)

    # Проверяем sendMessage payload
    send_digest_calls = [c for c in mock_delivery._calls if c[0] == "sendMessage"]
    assert len(send_digest_calls) > 0

    # Клавиатура должна быть: строки с кнопками 👍/👎, БЕЗ URL
    for method, payload in send_digest_calls:
        reply_markup = payload.get("reply_markup")
        assert reply_markup is not None, "Ожидался reply_markup при digest_enabled=true"
        keyboard = reply_markup.get("inline_keyboard", [])
        for row in keyboard:
            btn_urls = [b for b in row if "url" in b]
            assert len(btn_urls) == 0, (
                f"В строке дайджеста найдена URL-кнопка: {btn_urls} — "
                f"URL НЕ должен быть кнопкой (он в тексте)"
            )
            # Должны быть только 👍/👎
            texts = [b.get("text") for b in row]
            assert set(texts) <= {"👍", "👎"}, (
                f"Неподдерживаемые кнопки в строке: {texts}"
            )


# =============================================================================
# Тест 3: пуш-путь — send_batch добавляет 👍/👎 (не изменился)
# =============================================================================

def test_push_feedback_unchanged(tmp_path):
    """send_batch для пушей по-прежнему добавляет 👍/👎 кнопки в reply_markup."""
    mock_delivery = _MockTelegramDelivery()

    item = _mk_item(title="Push News")
    setattr(item, "feedback_target_id", 42)

    delivered = mock_delivery.send_batch([item])

    assert len(delivered) == 1
    calls = mock_delivery._calls
    send_calls = [c for c in calls if c[0] == "sendMessage"]
    assert len(send_calls) == 1

    payload = send_calls[0][1]
    reply_markup = payload.get("reply_markup")
    assert reply_markup is not None
    keyboard = reply_markup.get("inline_keyboard", [])
    all_btns = [b for row in keyboard for b in row]
    texts = [b["text"] for b in all_btns]
    # Строка 1: URL-кнопка; Строка 2: 👍/👎
    assert any("👍" in t for t in texts), f"👍 не найден в текстах кнопок: {texts}"
    assert any("👎" in t for t in texts), f"👎 не найден в текстах кнопок: {texts}"
    assert any("url" in b for b in all_btns), "URL-кнопка должна быть в пуш-сообщении"


# =============================================================================
# Тест 4: конфиг — digest_enabled парсится как bool, дефолт false
# =============================================================================

def test_config_digest_enabled_default_false():
    """DEFAULT_CONFIG: feedback.digest_enabled отсутствует → None/False → не создавать."""
    from newsbot.config import DEFAULT_CONFIG
    fb = DEFAULT_CONFIG.get("feedback", {})
    assert "digest_enabled" in fb, "digest_enabled должен быть в DEFAULT_CONFIG.feedback"
    assert fb["digest_enabled"] is False, "digest_enabled должен быть False по умолчанию"


def test_config_deep_merge_digest_enabled_true():
    """deep-merge: digest_enabled=true корректно проходит в финальный конфиг."""
    from newsbot.config import _deep_merge
    base = {"feedback": {"enabled": True, "digest_enabled": False}}
    overlay = {"feedback": {"digest_enabled": True}}
    result = _deep_merge(base, overlay)
    assert result["feedback"]["digest_enabled"] is True


def test_config_parse_bool_digest_enabled():
    """
    Интеграционный тест: pipeline правильно парсит digest_enabled из конфига.
    YAML-значения "false"/"true" (строки) должны давать тот же результат,
    что и булевы False/True.

    Проверяет РЕАЛЬНЫЙ путь: should_feedback в run_digest через _parse_bool.
    """
    from newsbot.pipeline import NewsPipeline
    from unittest.mock import patch

    # Мок-доставка, запоминающая вызовы
    calls_log: list = []

    class _CapturingDelivery(_MockTelegramDelivery):
        def send_text(self, text):
            calls_log.append(("send_text", text))
            return True

        def send_digest(self, text, item_buttons):
            calls_log.append(("send_digest", text, item_buttons))
            return True, []

    item1 = _mk_item(title="News 1")
    item2 = _mk_item(title="News 2")

    for digest_enabled_str, expect_buttons in [("false", False), ("true", True)]:
        calls_log.clear()
        mock_storage = MagicMock(spec=Storage)
        mock_storage.get_deferred_items.return_value = [item1, item2]
        mock_storage.get_meta.return_value = None
        mock_storage.get_or_create_feedback_target_from_item = MagicMock(
            side_effect=[10, 11]
        )

        cfg = {
            "sources": [],
            "database": {"path": ":memory:"},
            "telegram": {"token": "x", "chat_id": "x"},
            "llm": {"enabled": False},
            "digest": {"enabled": True},
            "feedback": {
                "enabled": True,
                "digest_enabled": digest_enabled_str,  # строковое значение!
            },
        }

        class PatchedPipeline(NewsPipeline):
            def __init__(self, config, storage, delivery):
                self.config = config
                self.storage = storage
                self.delivery = delivery
                self.sources = config.get("sources", []) or []
                self.llm_filter = None
                self._feedback_context = None
                self._llm_dedup_skipped_urls = set()
                self._max_retries = 3
                self._llm_calls_left = 150
                self._max_per_cycle = 150
                self._off_peak_only = False

        delivery = _CapturingDelivery()
        pipeline = PatchedPipeline(cfg, mock_storage, delivery)
        pipeline.run_digest(force=True)

        if expect_buttons:
            send_digest_calls = [c for c in calls_log if c[0] == "send_digest"]
            assert len(send_digest_calls) > 0, (
                f"digest_enabled={digest_enabled_str!r}: "
                f"ожидался send_digest с кнопками, вызовы={calls_log}"
            )
            # Кнопки должны быть с target_id (без "url")
            _, _, item_buttons = send_digest_calls[0]
            assert len(item_buttons) == 2, "Ожидались 2 строки с кнопками"
            assert all("target_id" in b for b in item_buttons), "Каждая кнопка должна иметь target_id"
            assert all("url" not in b for b in item_buttons), "Кнопки НЕ должны содержать 'url'"
        else:
            send_digest_calls = [c for c in calls_log if c[0] == "send_digest"]
            assert len(send_digest_calls) == 0, (
                f"digest_enabled={digest_enabled_str!r}: "
                f"send_digest НЕ должен вызываться, вызовы={calls_log}"
            )


# =============================================================================
# Тест 5: digest_enabled=false + feedback.enabled=false → тоже send_text
# =============================================================================

def test_digest_feedback_disabled_send_text(tmp_path):
    """При feedback.enabled=false дайджест отправляется без клавиатуры."""
    mock_delivery = _MockTelegramDelivery()

    cfg = {
        "sources": [],
        "database": {"path": str(tmp_path / "test.db")},
        "telegram": {"token": "x", "chat_id": "x"},
        "llm": {"enabled": False},
        "digest": {"enabled": True},
        "feedback": {
            "enabled": False,  # фидбэк выключен полностью
            "digest_enabled": True,  # ← включено, но общий enabled=false
        },
    }

    mock_storage = MagicMock(spec=Storage)
    mock_storage.get_deferred_items.return_value = [_mk_item(title="X")]
    mock_storage.get_meta.return_value = None

    class PatchedPipeline(NewsPipeline):
        def __init__(self, config, storage, delivery):
            self.config = config
            self.storage = storage
            self.delivery = delivery
            self.sources = config.get("sources", []) or []
            self.llm_filter = None
            self._feedback_context = None
            self._llm_dedup_skipped_urls = set()
            self._max_retries = 3
            self._llm_calls_left = 150
            self._max_per_cycle = 150
            self._off_peak_only = False

    pipeline = PatchedPipeline(cfg, mock_storage, mock_delivery)
    pipeline.run_digest(force=True)

    calls = mock_delivery._calls
    send_calls = [c for c in calls if c[0] == "sendMessage"]
    reply_markups = [c[1].get("reply_markup") for c in send_calls]
    # send_text — без reply_markup
    assert all(r is None for r in reply_markups), (
        f"При feedback.enabled=false ожидался send_text (без reply_markup): {reply_markups}"
    )