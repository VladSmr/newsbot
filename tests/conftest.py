"""
conftest.py — pytest configuration.

Добавляет корень проекта в sys.path, чтобы тесты
находили пакет `newsbot` при запуске `pytest` или `python -m pytest`
из корня проекта.
"""
import sys
from pathlib import Path

import pytest

# проектный корень: tests/ → корень
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────
# P0 (аудит): блокировка непредвиденных реальных HTTP-запросов.
# Патчим httpx.HTTPTransport.handle_request на уровне транспорта —
# если тест вызывает httpx.get/post без MockTransport или
# patch.object → патч поднимает RuntimeError, и тест падает
# немедленно, а не «случайно зелёным» от таймаута.
#
# Почему httpx.HTTPTransport.handle_request, а не socket:
#   - socket-патч на Windows затрагивает create_connection, а не только
#     socket() — это может сломать MockTransport (httpcore).
#   - handle_request покрывает ВСЕ high-level HTTP-вызовы (get, post, …)
#     и НЕ затрагивает httpx.MockTransport, который подменяет ОБА
#     synchronous и asynchronous handle_request (работает на том же уровне).
#   - patch.object(client, "post") работает на экземпляре (другой namespace),
#     поэтому тесты с собственной мок-логикой НЕ затрагиваются.
# ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch):
    """
    autouse — действует на все тесты без дополнительного импорта.
    Патчит httpx.HTTPTransport.handle_request на уровне транспорта.
    Это НЕ затрагивает:
      - patch("httpx.Client") — заменяет весь класс Client
      - httpx.MockTransport — работает на уровне transport.handle_request()
        но подменяет ОБА synchronous и asynchronous handle_request
      - patch.object(client, "post") — патчит экземпляр, не класс
    """
    import httpx

    _original_handle = httpx.HTTPTransport.handle_request

    def _blocked_handle_request(self, request):
        raise RuntimeError(
            "REAL NETWORK CALL BLOCKED by conftest.py: "
            "httpx.HTTPTransport.handle_request вызван без MockTransport. "
            "Используйте httpx.MockTransport или patch.object(..., side_effect=...) "
            "для моков HTTP-вызовов."
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request",
                      _blocked_handle_request, raising=False)
    yield
    # restore не нужен (monkeypatch сам восстанавливает)