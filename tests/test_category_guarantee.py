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