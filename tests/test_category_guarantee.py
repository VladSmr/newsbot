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
