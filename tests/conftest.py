"""
conftest.py — pytest configuration.

Добавляет корень проекта в sys.path, чтобы тесты
находили пакет `newsbot` при запуске `pytest` или `python -m pytest`
из корня проекта.
"""
import sys
from pathlib import Path

# проектный корень: tests/ → корень
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))