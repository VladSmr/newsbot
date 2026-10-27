"""Тесты гарантии минимума по категориям."""
import sys
from pathlib import Path

# корневой путь проекта для импорта newsbot
sys.path.insert(0, str(Path(__file__).parent.parent))

from newsbot.pipeline import _category_guarantee, _infer_category


class FakeItem:
    def __init__(self, score, source, tags=None):
        self.llm_score = score
        self.source = source
        self.tags = tags


def test_guarantee_cinema_min():
    items = [
        FakeItem(3, 'deadline'),   # cinema
        FakeItem(2, 'deadline'),   # cinema
        FakeItem(5, 'tass'),        # other
        FakeItem(4, 'tass'),        # other
        FakeItem(1, 'deadline'),   # cinema
    ]
    result = _category_guarantee(items, {'cinema': 2})
    cats = [_infer_category(i) for i in result]
    assert cats.count('cinema') >= 2, f"Нужно >=2 cinema, получили {cats.count('cinema')}"


def test_guarantee_empty():
    assert _category_guarantee([], {'cinema': 2}) == []


def test_guarantee_not_enough_items():
    items = [FakeItem(3, 'deadline')]  # только 1 cinema
    result = _category_guarantee(items, {'cinema': 2})
    assert len(result) == 1  # меньше чем min, но больше нечего добавить


def test_infer_category():
    assert _infer_category(FakeItem(5, 'deadline')) == 'cinema'
    assert _infer_category(FakeItem(5, 'ign')) == 'games'
    assert _infer_category(FakeItem(5, 'spacenews')) == 'space'
    assert _infer_category(FakeItem(5, 'tass')) == 'other'
    assert _infer_category(FakeItem(5, 'tass', '["film","movie"]')) == 'cinema'


def test_infer_category_naked_science():
    """P2: "Naked Science" нормализуется в "nakedscience" (пробел убирается),
    а в алиасах стоял 'naked science' — не совпадало никогда."""
    for source in ('Naked Science', 'naked science', 'NakedScience'):
        assert _infer_category(FakeItem(5, source)) == 'science', source


def test_all_aliases_are_normalized_keys():
    """Алиасы должны быть в нормализованном виде (без пробелов/пунктуации)."""
    from newsbot.pipeline import _SOURCE_ALIASES
    import re
    for key in _SOURCE_ALIASES:
        normalized = re.sub(r"[^\w]", "", key.lower(), flags=re.UNICODE)
        assert normalized == key, f"алиас {key!r} не совпадёт после нормализации"


# ---------------------------------------------------------------------------
# Minor 1 (wave 2): строковые значения category_min из YAML
# ---------------------------------------------------------------------------

def test_string_min_count_is_coerced():
    """category_min: {'cinema': '2'} — гарантия выполняется, не TypeError."""
    items = [
        FakeItem(1, 'deadline'),  # cinema — единственная, ниже min
        FakeItem(5, 'tass'),      # other
        FakeItem(4, 'tass'),      # other
    ]
    pool = [FakeItem(9, 'deadline')]  # лучший cinema в пуле
    result = _category_guarantee(items, {'cinema': '2'}, pool=pool)
    cats = [_infer_category(i) for i in result]
    assert cats.count('cinema') >= 2, f"Нужно >=2 cinema, получили {cats}"


def test_invalid_min_count_ignored_without_crash():
    """Нечисловой min ('abc') → категория игнорируется (0), без исключения."""
    items = [FakeItem(5, 'tass'), FakeItem(3, 'deadline')]
    result = _category_guarantee(items, {'cinema': 'abc'})
    assert [i is it for i, it in zip(result, items)] == [True, True]


def test_non_dict_category_min_ignored():
    """category_min не-dict (опечатка в YAML) → no-op, без exception."""
    items = [FakeItem(5, 'tass')]
    assert _category_guarantee(items, ['cinema']) == items


# ---------------------------------------------------------------------------
# Nit (wave 4): _coerce_int выровнен с dedup._cfg_int
# ---------------------------------------------------------------------------

def test_float_string_min_count_is_coerced():
    """'2.0' (float-строка из YAML) раньше отвергалась → гарантия 0.

    Было: int("2.0") → ValueError → warning + default 0, категория молча
    переставала гарантироваться.
    """
    items = [
        FakeItem(1, 'deadline'),  # cinema — единственная, ниже min
        FakeItem(5, 'tass'),
        FakeItem(4, 'tass'),
    ]
    pool = [FakeItem(9, 'deadline')]
    result = _category_guarantee(items, {'cinema': '2.0'}, pool=pool)
    cats = [_infer_category(i) for i in result]
    assert cats.count('cinema') >= 2, f"Нужно >=2 cinema, получили {cats}"


def test_float_min_count_is_floored_not_truncated_silently():
    """2.9 → floor 2 (нецелый float не усекается молча — warning)."""
    import logging

    items = [
        FakeItem(1, 'deadline'),
        FakeItem(5, 'tass'),
        FakeItem(4, 'tass'),
    ]
    pool = [FakeItem(9, 'deadline')]
    logger = logging.getLogger("newsbot.pipeline")
    records = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record)

    h = _H()
    logger.addHandler(h)
    try:
        result = _category_guarantee(items, {'cinema': 2.9}, pool=pool)
    finally:
        logger.removeHandler(h)
    cats = [_infer_category(i) for i in result]
    assert cats.count('cinema') == 2, f"2.9 → 2 cinema, получили {cats}"
    assert any("нецелое число" in r.getMessage() for r in records), \
        [r.getMessage() for r in records]


def test_nan_min_count_falls_back_to_zero_without_crash():
    """NaN (YAML `.nan`) → 0 + warning, не ValueError из int()."""
    items = [FakeItem(5, 'tass'), FakeItem(3, 'deadline')]
    result = _category_guarantee(items, {'cinema': float('nan')})
    assert result == items


# =============================================================================
# F4-тесты: content-override для domain-specific источников
# =============================================================================

class FakeItemWithText:
    """FakeItem с title и summary для F4-тестов."""
    def __init__(self, score, source, tags=None, title="", summary=""):
        self.llm_score = score
        self.source = source
        self.tags = tags
        self.title = title
        self.summary = summary


def test_variety_political_becomes_other():
    """F4: Variety + политический title → 'other' (не 'cinema').

    Кейс из реального дайджеста: «Lula's Election Win…» (Variety) →
    политика, должна попасть в 📰 Основные новости.
    """
    item = FakeItemWithText(
        score=7,
        source='Variety',
        title="Lula's Election Win: How Taylor Swift's Advice Shaped the Result",
        summary="Brazil's president election aftermath analysis.",
    )
    assert _infer_category(item) == 'other', \
        "Variety + politics → 'other', not 'cinema'"


def test_variety_film_stays_cinema():
    """F4: Variety + film markers → 'cinema' (content-override не срабатывает)."""
    item = FakeItemWithText(
        score=7,
        source='Variety',
        title="New Marvel Movie Trailer Drops — First Look at Phase 6",
        summary="The latest Marvel cinematic universe preview.",
    )
    assert _infer_category(item) == 'cinema', \
        "Variety + film markers → 'cinema'"


def test_deadline_political_becomes_other():
    """F4: Deadline + политический title → 'other'."""
    item = FakeItemWithText(
        score=7,
        source='Deadline',
        title="Congress Passes New Sanctions Bill Against Russia",
        summary="Legislative update on international sanctions.",
    )
    assert _infer_category(item) == 'other', \
        "Deadline + political → 'other'"


def test_deadline_oscar_stays_cinema():
    """F4: Deadline + oscars/cinema → 'cinema'."""
    item = FakeItemWithText(
        score=7,
        source='Deadline',
        title="Oscar Nominations Announced — Full List of 2026 Nominees",
        summary="Academy Awards nominations reveal.",
    )
    assert _infer_category(item) == 'cinema', \
        "Deadline + oscars → 'cinema'"


def test_ign_political_becomes_other():
    """F4: IGN + war/conflict → 'other'."""
    item = FakeItemWithText(
        score=7,
        source='IGN',
        title="NATO Summit: New War Strategy Announced",
        summary="Military alliance decisions from Brussels summit.",
    )
    assert _infer_category(item) == 'other', \
        "IGN + war/politics → 'other'"


def test_ign_gaming_stays_games():
    """F4: IGN + gaming → 'games'."""
    item = FakeItemWithText(
        score=7,
        source='IGN',
        title="PlayStation 6 Announced — Release Date and Features",
        summary="Sony's next-generation console reveal.",
    )
    assert _infer_category(item) == 'games', \
        "IGN + gaming → 'games'"


def test_nasa_political_becomes_other():
    """F4: NASA + political → 'other'."""
    item = FakeItemWithText(
        score=7,
        source='NASA',
        title="Congress Cuts NASA Budget Amid Election-Year Spending Debate",
        summary="Legislative budget reduction impact on space program.",
    )
    assert _infer_category(item) == 'other', \
        "NASA + politics → 'other'"


def test_nasa_space_stays_space():
    """F4: NASA + space markers → 'space'."""
    item = FakeItemWithText(
        score=7,
        source='NASA',
        title="SpaceX Rocket Launch Successful — ISS Resupply Mission",
        summary="Cargo delivery to international space station.",
    )
    assert _infer_category(item) == 'space', \
        "NASA + space → 'space'"


def test_russian_political_markers():
    """F4: русские политические маркеры в title/summary переопределяют категорию."""
    item = FakeItemWithText(
        score=7,
        source='Variety',
        title="Президентские выборы 2026: основные кандидаты",
        summary="Обзор предвыборной гонки в Бразилии.",
    )
    assert _infer_category(item) == 'other', \
        "Russian political markers → 'other'"


def test_generic_source_political_unaffected():
    """F4: generic источник (tass) с политическим контентом → 'other' (по tags)."""
    item = FakeItemWithText(
        score=7,
        source='tass',
        title="Government Announces New Tariff Policy",
        summary="Economic policy changes announced today.",
        tags='["politics"]',
    )
    # Без source-first override (tass → other), результат зависит от tags.
    # Нет профильных tags → 'other' (по tags или fallthrough).
    assert _infer_category(item) == 'other'
