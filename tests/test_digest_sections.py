"""
Тесты группировки дайджеста по категориям (v1.3.0).

Покрывает:
  - фиксированный порядок секций и их заголовки
  - пустые секции не выводятся, секция из 1 новости выводится
  - сортировка внутри секции — прежний порядок (score DESC)
  - бэклог/пейджинг не рвёт секцию, если она влезает в сообщение
  - бюджет 4096 приоритетнее: огромная секция рвётся (как и раньше)
  - кнопки 👍/👎 соответствуют своим items после перегруппировки
    (кнопки строятся по url-карте, порядок = порядок items в сообщении)
  - пересборка groups (pipeline: выбивание item'ов) сохраняет
    соответствие «кнопка ↔ новость»

Запуск: python -m pytest tests/test_digest_sections.py -v
"""
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root))

from newsbot.collector import NewsItem
from newsbot.digest import (
    SECTION_TITLES,
    _TG_MESSAGE_MAX,
    _send_digest_with_targets,
    build_digest_messages,
    infer_category,
)


# =============================================================================
# helpers
# =============================================================================

def _u16(s: str) -> int:
    """Длина строки в UTF-16 code units — ровно так лимит считает Telegram."""
    return len(s.encode("utf-16-le")) // 2


def _mk_item(source="tass", title="Test Title", url=None, summary="Body.",
             llm_score=6, tags=None) -> NewsItem:
    if url is None:
        url = f"https://example.com/{title}"
    return NewsItem(
        source=source, url=url, title=title, summary=summary,
        published="2025-01-01 12:00", published_parsed=None,
        llm_score=llm_score, tags=tags,
    )


def _section_titles_in(msg: str):
    """Заголовки секций (в порядке появления) в тексте сообщения."""
    return [title for _, title in SECTION_TITLES
            if f"<b>{title}</b>" in msg]


def _section_order(msg: str):
    """Индексы позиций заголовков секций — для проверки порядка."""
    positions = []
    for idx, (_, title) in enumerate(SECTION_TITLES):
        pos = msg.find(f"<b>{title}</b>")
        if pos >= 0:
            positions.append((pos, idx))
    return [idx for _, idx in sorted(positions)]


# =============================================================================
# 1. Порядок и заголовки секций
# =============================================================================

def test_sections_order_and_titles():
    """Секции идут в фиксированном порядке с правильными заголовками."""
    items = [
        _mk_item("krebsonsecurity", "cyber-1", llm_score=9),
        _mk_item("ign", "games-1", llm_score=8),
        _mk_item("deadline", "cinema-1", llm_score=7),
        _mk_item("habr", "tech-1", llm_score=6),
        _mk_item("spacenews", "space-1", llm_score=5),
        _mk_item("sciencedaily", "science-1", llm_score=4),
        _mk_item("tass", "general-1", llm_score=3),
    ]
    groups, included, failed = build_digest_messages(items)

    assert len(groups) == 1
    msg, msg_items = groups[0]
    assert failed == []

    # порядок секций — как в SECTION_TITLES (general, science, space, tech,
    # cinema, games, cyber)
    assert _section_order(msg) == list(range(len(SECTION_TITLES)))
    assert _section_titles_in(msg) == [t for _, t in SECTION_TITLES]

    # порядок items в группе соответствует порядку блоков в тексте
    assert [it.title for it in msg_items] == [
        "general-1", "science-1", "space-1", "tech-1",
        "cinema-1", "games-1", "cyber-1",
    ]
    assert len(included) == len(items)


def test_empty_sections_not_rendered():
    """Пустые секции отсутствуют в тексте."""
    items = [
        _mk_item("tass", "general-1", llm_score=7),
        _mk_item("tass", "general-2", llm_score=6),
        _mk_item("ign", "games-1", llm_score=5),
    ]
    groups, _, _ = build_digest_messages(items)
    msg = groups[0][0]

    assert "<b>📰 Основные новости</b>" in msg
    assert "<b>🎮 Игры</b>" in msg
    for absent in ("🔬 Наука", "🚀 Космос", "💻 Технологии",
                   "🎬 Кино", "🔐 Кибербезопасность"):
        assert absent not in msg, f"пустая секция '{absent}' не должна выводиться"


def test_single_item_section_rendered():
    """Секция из одной новости показывается (заголовок + новость)."""
    items = [
        _mk_item("tass", "general-1", llm_score=7),
        _mk_item("tass", "general-2", llm_score=6),
        _mk_item("krebsonsecurity", "cyber-only", llm_score=5),
    ]
    groups, _, _ = build_digest_messages(items)
    msg = groups[0][0]

    assert "<b>🔐 Кибербезопасность</b>" in msg
    assert "cyber-only" in msg


def test_detail_order_within_section_preserved():
    """Внутри секции — исходный порядок (score DESC после финального сорта)."""
    items = [
        _mk_item("sciencedaily", "science-high", llm_score=9),
        _mk_item("tass", "general-mid", llm_score=8),
        _mk_item("sciencedaily", "science-low", llm_score=3),
    ]
    groups, _, _ = build_digest_messages(items)
    msg, msg_items = groups[0]

    assert [it.title for it in msg_items] == [
        "general-mid", "science-high", "science-low",
    ]
    # порядок внутри секции сохранён
    assert msg.index("science-high") < msg.index("science-low")


def test_unknown_category_falls_back_to_general():
    """Категория 'other' (неизвестный source) → секция «Основные новости»."""
    item = _mk_item("SomeUnknownSource", "unknown-1", llm_score=5)
    assert infer_category(item) == "general"

    groups, _, _ = build_digest_messages([item])
    msg = groups[0][0]
    assert "<b>📰 Основные новости</b>" in msg
    assert msg.count("<b>📰 Основные новости</b>") == 1


# =============================================================================
# 2. Пейджинг и целостность секций
# =============================================================================

def test_pagination_does_not_split_section_that_fits():
    """
    Секция, влезающая в пустое сообщение, не рвётся между сообщениями.

    Данные подобраны так, что суммарный объём > 4096 (без переноса science
    не помещается), но сама science-секция в пустое сообщение влезает —
    значит обязана сработать ветка атомарного переноса секции
    (digest.py:394-399). При её удалении science разорвётся на 2 сообщения
    и тест упадёт.
    """
    # general: 5 «тяжёлых» новостей (summary 500 — потолок рендера),
    # science: 3 средних, целиком влезающих в пустое сообщение.
    items = [
        _mk_item("tass", f"general-{i}", summary="g" * 500, llm_score=6)
        for i in range(5)
    ] + [
        _mk_item("sciencedaily", f"science-{i}", summary="s" * 400, llm_score=5)
        for i in range(3)
    ]
    groups, included, _ = build_digest_messages(items, max_items_per_message=20)

    # фича реально нагружена: суммарно текст НЕ влезает в одно сообщение
    assert sum(len(msg) for msg, _ in groups) > _TG_MESSAGE_MAX

    # каждая новость ровно в одном сообщении
    titles = [it.title for _, grp in groups for it in grp]
    assert sorted(titles) == sorted(it.title for it in included)
    assert len(titles) == len(set(titles)) == 8

    # general уместился в первое, science перенесена целиком во второе
    assert len(groups) == 2
    assert [it.title for it in groups[0][1]] == [f"general-{i}" for i in range(5)]

    # секция science целиком в одном сообщении и заголовок есть ровно 1 раз
    sci_msgs = [msg for msg, grp in groups
                if any(it.title.startswith("science-") for it in grp)]
    assert len(sci_msgs) == 1
    assert sci_msgs[0].count("<b>🔬 Наука</b>") == 1
    assert sum(1 for it in groups[0][1] if it.title.startswith("science-")) + \
        sum(1 for _, grp in groups[1:] for it in grp
            if it.title.startswith("science-")) == 3



def test_pagination_repeats_section_header_when_section_split():
    """Огромная секция рвётся бюджетом — каждое сообщение с заголовком секции."""
    items = [
        _mk_item("tass", f"general-{i}", summary="x" * 600, llm_score=6)
        for i in range(15)
    ]
    groups, _, _ = build_digest_messages(items, max_items_per_message=20)

    assert len(groups) >= 2
    for msg, grp in groups:
        assert _u16(msg) <= _TG_MESSAGE_MAX, f"превышен лимит 4096 units: {_u16(msg)}"
        assert "<b>📰 Основные новости</b>" in msg
        # заголовок секции повторяется, но не дублируется внутри сообщения
        assert msg.count("<b>📰 Основные новости</b>") == 1
        assert len(grp) > 0

    # нумерация «N–M из TOTAL» непрерывна
    assert "— 1–" in groups[0][0]
    total = 15
    assert f"из {total} новостей" in groups[0][0]
    last = groups[-1][0]
    assert f"из {total} новостей" in last
    assert max(_u16(msg) for msg, _ in groups) <= _TG_MESSAGE_MAX


def test_budget_priority_over_section_integrity():
    """
    Бюджет 4096 приоритетнее целостности секции: если секция целиком
    не влезает в сообщение — она рвётся, каждый кусок с заголовком.

    Данные — 6 реально больших entries (title ~250 после обрезки в
    _render_item_raw + body 500 = потолки рендера), суммарно >> 4096.
    Т.е. «огромный item» здесь реально большой ПОСЛЕ рендера, а не
    усечённый до 500 summary.
    """
    items = [
        _mk_item("tass", f"T{i}" + "x" * 248, summary="z" * 500, llm_score=7)
        for i in range(6)
    ]
    groups, included, failed = build_digest_messages(items)

    assert failed == []
    assert len(included) == 6

    # секция не влезает целиком → бюджет её разорвал
    assert sum(len(msg) for msg, _ in groups) > _TG_MESSAGE_MAX
    assert len(groups) == 2

    for msg, grp in groups:
        assert _u16(msg) <= _TG_MESSAGE_MAX, f"превышен лимит 4096 units: {_u16(msg)}"
        assert len(grp) > 0
        assert "<b>📰 Основные новости</b>" in msg
        # заголовок повторяется в каждом сообщении, но не дублируется внутри
        assert msg.count("<b>📰 Основные новости</b>") == 1
        # рвём по бюджету, а не урезанием entry: тела не обрезаны
        assert "…" not in msg

    # все новости разложены, каждая ровно один раз
    titles = [it.title for _, grp in groups for it in grp]
    assert sorted(titles) == sorted(it.title for it in items)



def _parse_msg_range(msg: str, total: int):
    """Возвращает (start, end) из заголовка «… — S–E из TOTAL новостей …»."""
    m = re.search(r"— (\d+)–(\d+) из (\d+) новостей", msg)
    if m:
        start, end, seen_total = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.search(r"— (\d+) из (\d+) новостей", msg)
        assert m, f"не найден номерной заголовок в сообщении: {msg[:80]!r}"
        start = end = int(m.group(1))
        seen_total = int(m.group(2))
    assert seen_total == total
    return start, end


def test_multi_message_numbering_continuous():
    """Нумерация «N–M из TOTAL» непрерывна при переносах секций."""
    items = [
        _mk_item("tass", f"g{i}", summary="a" * 400, llm_score=7)
        for i in range(4)
    ] + [
        _mk_item("sciencedaily", f"s{i}", summary="b" * 700, llm_score=6)
        for i in range(4)
    ]
    groups, _, _ = build_digest_messages(items)
    total = len(items)
    assert len(groups) >= 2

    seen = 0
    ranges = []
    for msg, grp in groups:
        assert f"из {total} новостей" in msg
        start, end = _parse_msg_range(msg, total)
        # диапазон сообщения совпадает с числом items и продолжает предыдущий
        assert end - start + 1 == len(grp)
        assert start == seen + 1, f"разрыв нумерации: {start} после {seen}"
        seen = end
        ranges.append((start, end))

    # сплошное покрытие 1..TOTAL и последняя строка «TOTAL из TOTAL»
    assert seen == total
    assert ranges[0][0] == 1
    assert ranges[-1][1] == total
    assert f"— {total} из {total} новостей" in groups[-1][0] or \
        f"— {ranges[-1][0]}–{total} из {total} новостей" in groups[-1][0]


def test_multi_message_numbering_continuous_intra_section_split():
    """
    Разрыв ОДНОЙ секции на несколько сообщений (intra-section split).

    Заголовок секции повторяется в каждом куске (digest.py:412-429),
    а нумерация «N–M из TOTAL» остаётся сплошной: строго [1..k][k+1..n],
    без дыр и пересечений; диапазон совпадает с числом items в куске.
    """
    for n in (12, 16, 20):
        items = [
            _mk_item("tass", f"g{i}", summary="a" * 500, llm_score=7)
            for i in range(n)
        ]
        groups, included, failed = build_digest_messages(
            items, max_items_per_message=20
        )

        assert failed == [], f"n={n}"
        assert len(included) == n
        # сценарий реально нагружен: одна секция разорвана бюджетом
        assert len(groups) >= 2, f"n={n}: секция не разорвана"

        ranges = []
        for msg, grp in groups:
            assert _u16(msg) <= _TG_MESSAGE_MAX, f"n={n}: превышен лимит 4096 units"
            # секция одна → её заголовок есть в каждом куске ровно один раз
            assert msg.count("<b>📰 Основные новости</b>") == 1, f"n={n}"
            start, end = _parse_msg_range(msg, n)
            assert end - start + 1 == len(grp), f"n={n}: {start}-{end} vs {len(grp)}"
            ranges.append((start, end))

        # строго [1..k][k+1..n]
        assert ranges[0][0] == 1, f"n={n}"
        assert ranges[-1][1] == n, f"n={n}"
        for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:]):
            assert next_start == prev_end + 1, \
                f"n={n}: разрыв нумерации {prev_end} → {next_start}"

        # все новости разложены, каждая ровно один раз
        titles = [it.title for _, grp in groups for it in grp]
        assert sorted(titles) == sorted(f"g{i}" for i in range(n)), f"n={n}"



# =============================================================================
# 3. Кнопки фидбэка: соответствие item'ам после перегруппировки
# =============================================================================

def _buttons_by_url_map(groups, target_id_by_url):
    """Повторяет логику pipeline: кнопки строятся по url-карте групп."""
    return [
        [{"target_id": target_id_by_url.get(it.url)} for it in grp]
        for _, grp in groups
    ]


def test_buttons_match_items_after_grouping():
    """Кнопки 👍/👎 соответствуют своим items (по url, не по индексам)."""
    items = [
        _mk_item("tass", "general-1", llm_score=9),
        _mk_item("ign", "games-1", llm_score=8),
        _mk_item("tass", "general-2", llm_score=7),
        _mk_item("krebsonsecurity", "cyber-1", llm_score=6),
        _mk_item("sciencedaily", "science-1", llm_score=5),
    ]
    # target_id уникален для url
    target_id_by_url = {it.url: 1000 + i for i, it in enumerate(items)}

    groups, _, _ = build_digest_messages(items)
    buttons = _buttons_by_url_map(groups, target_id_by_url)

    for (msg, grp), btns in zip(groups, buttons):
        assert len(btns) == len(grp)
        for item, btn in zip(grp, btns):
            assert btn["target_id"] == target_id_by_url[item.url]
        # порядок появления в тексте совпадает с порядком items группы
        positions = [msg.index(it.title) for it in grp]
        assert positions == sorted(positions)


def test_buttons_match_items_after_pipeline_rebuild():
    """
    Пересборка groups после выбивания item'ов (pipeline.py:1821-1866)
    сохраняет соответствие кнопок и новостей (кнопки — по url-карте).
    """
    items = [
        _mk_item("tass", "general-1", llm_score=9),
        _mk_item("ign", "games-1", llm_score=8),
        _mk_item("tass", "general-2", llm_score=7),
        _mk_item("krebsonsecurity", "cyber-1", llm_score=6),
        _mk_item("sciencedaily", "science-1", llm_score=5),
    ]
    target_id_by_url = {it.url: 1000 + i for i, it in enumerate(items)}

    groups, _, _ = build_digest_messages(items, max_items_per_message=20)

    # выбиваем games-1 (упавший feedback-target)
    failed_url = next(it.url for it in items if it.title == "games-1")
    failed_set = {failed_url}
    surviving = [it for _, grp in groups for it in grp
                 if it.url not in failed_set]

    groups2, _, _ = build_digest_messages(surviving, max_items_per_message=20)
    buttons2 = _buttons_by_url_map(groups2, target_id_by_url)

    sent_titles = []
    for (msg, grp), btns in zip(groups2, buttons2):
        assert len(btns) == len(grp)
        assert "games-1" not in msg
        assert f"из {len(surviving)} новостей" in msg
        for item, btn in zip(grp, btns):
            assert btn["target_id"] == target_id_by_url[item.url]
            sent_titles.append(item.title)

    assert "games-1" not in sent_titles
    assert len(sent_titles) == len(surviving)
    # порядок секций сохранён и после пересборки (секции games больше нет)
    games_idx = next(i for i, (key, _) in enumerate(SECTION_TITLES) if key == "games")
    assert _section_order(groups2[0][0]) == [
        i for i in _section_order(groups[0][0]) if i != games_idx
    ]


# =============================================================================
# 6. Транзиентный ImportError не запирает fallback навсегда
# =============================================================================

def test_transient_import_error_does_not_gate_forever(monkeypatch, caplog):
    """
    Флаг _INFER_CATEGORY_UNAVAILABLE — только глушитель warning'ов, а не
    гейт импорта: после восстановления импорта pipeline категории снова
    работают, флаг сбрасывается.
    """
    import logging
    import types

    import newsbot.digest as digest_mod

    monkeypatch.setattr(digest_mod, "_INFER_CATEGORY_UNAVAILABLE", False)
    # None в sys.modules → `from .pipeline import ...` бросает ImportError
    monkeypatch.setitem(sys.modules, "newsbot.pipeline", None)

    item = _mk_item("deadline", "cinema-1", llm_score=7)

    with caplog.at_level(logging.WARNING, logger="newsbot.digest"):
        assert infer_category(item) == "general"
        assert digest_mod._INFER_CATEGORY_UNAVAILABLE is True
        # повторный провал — без нового warning'а (флаг глушит шум)
        assert infer_category(item) == "general"
    assert len(caplog.records) == 1

    # импорт «восстановился» (транзиентный сбой закончился).
    # Подставляем лёгкий стаб вместо реального модуля, чтобы не
    # переисполнять newsbot.pipeline (иначе в sys.modules появился бы
    # дубликат модуля и другие тесты получили бы чужие объекты).
    monkeypatch.setitem(
        sys.modules,
        "newsbot.pipeline",
        types.SimpleNamespace(_infer_category=lambda _item: "cinema"),
    )

    assert infer_category(item) == "cinema"
    assert digest_mod._INFER_CATEGORY_UNAVAILABLE is False


# =============================================================================
# 7. Бюджет дайджеста — в UTF-16 code units (как считает Telegram)
# =============================================================================

def _emoji_overload_items(n: int = 20):
    """
    n новостей с эмодзи в заголовке и теле: code points << UTF-16 units.
    Каждый футер содержит 🔗 (не-BMP, 2 units) — как и реальные футеры
    дайджеста со ссылкой.
    """
    return [
        _mk_item(
            "tass",
            title=f"🔥 Новость {i} " + "🔥" * 5,
            summary="😀" * 400,
            url=f"https://example.com/news/{i}",
            llm_score=6,
        )
        for i in range(n)
    ]


def test_budget_counts_utf16_units_not_code_points():
    """
    Регресс (Critical): страница с ≥20 🔗-футерами и эмодзи-заголовками.

    Раньше все оценки размера считались в code points (len), поэтому
    страница ~4076 cp с не-BMP символами реально занимала 4100+ UTF-16
    units → Telegram 400 «message is too long» → send_digest=False →
    items оставались deferred и повторялись каждое окно до purge.

    Проверяем: каждая часть ≤ 4096 UTF-16 units (len(utf-16-le)//2).
    """
    items = _emoji_overload_items(20)
    groups, included, failed = build_digest_messages(
        items, max_items_per_message=20
    )

    assert failed == []
    assert len(included) == 20

    # фича нагружена: суммарно НЕ влезает в одно сообщение
    assert len(groups) >= 2
    # все 🔗-футеры на месте (≥20 не-BMP символов в датасете)
    assert sum(msg.count("🔗") for msg, _ in groups) == 20

    for msg, grp in groups:
        units = _u16(msg)
        assert units <= _TG_MESSAGE_MAX, (
            f"часть {units} UTF-16 units > 4096 "
            f"(code points: {len(msg)}, items: {len(grp)})"
        )

    # эмодзи реально «удваивают» размер: units заметно больше code points
    assert sum(_u16(m) for m, _ in groups) > sum(len(m) for m, _ in groups)

    # ни одна новость не потеряна и не продублирована
    titles = [it.title for _, grp in groups for it in grp]
    assert sorted(titles) == sorted(it.title for it in items)


def test_old_codepoint_budget_would_overflow():
    """
    Гард на сам тест: при старой оценке через len() (подменяем _utf16_units
    на len) тот же датасет даёт часть > 4096 units — т.е. предыдущий тест
    действительно ловит баг, а не проходит «по счастливой случайности».
    """
    items = _emoji_overload_items(20)
    with patch("newsbot.digest._utf16_units", new=len):
        groups, _, _ = build_digest_messages(items, max_items_per_message=20)
    assert max(_u16(m) for m, _ in groups) > _TG_MESSAGE_MAX


def test_send_digest_warning_uses_utf16_units(caplog):
    """
    Warning в _send_digest_with_targets тоже считает units: 4096 emoji
    = 8192 units при 4096 code points (старый check молчал).
    """
    import logging

    msg = "😀" * _TG_MESSAGE_MAX
    assert len(msg) == _TG_MESSAGE_MAX
    assert _u16(msg) > _TG_MESSAGE_MAX

    delivery = MagicMock()
    delivery.send_digest.return_value = True

    with caplog.at_level(logging.WARNING, logger="newsbot.digest"):
        sent_count, _, _ = _send_digest_with_targets(
            delivery, [(msg, [])], [[]]
        )

    assert sent_count == 0  # items нет — счётчик по items, не по сообщениям
    warnings = [r.getMessage() for r in caplog.records]
    assert any("UTF-16 units" in w for w in warnings), warnings
    assert any(str(_u16(msg)) in w for w in warnings), warnings
