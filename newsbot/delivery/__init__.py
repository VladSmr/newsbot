# delivery package
from datetime import datetime, timezone
from typing import Optional


# format_time_utc определён ПЕРЕД импортами .console и .telegram,
# чтобы разорвать циклический импорт:
#   console.py делает `from . import format_time_utc`
#   telegram.py делает `from . import format_time_utc`
# Если определить после импорта → AttributeError (имя ещё не существует).
def format_time_utc(dt: Optional[datetime], fallback: Optional[datetime] = None) -> str:
    """
    Форматирует datetime в локальное время.

    Используется и в telegram.py, и в console.py — единая точка.

    Параметры
    ----------
    dt : datetime
        Время для форматирования. Если None — используется fallback.
    fallback : datetime
        Резервное время (обычно fetched_at). Обязательно если dt=None.

    Поведение
    ---------
    - Naive datetime → считается UTC.
    - aware datetime → конвертируется в local через astimezone().
    - published_parsed (feedparser, time.struct_time) → конвертируется в UTC aware.
    - При любой ошибке → строковый формат "%d.%m.%Y %H:%M" по UTC.
    """
    if dt is None:
        dt = fallback
    if dt is None:
        return "—"

    # ── обработка feedparser time.struct_time ──────────────────────
    if hasattr(dt, "tm_year"):  # struct_time
        import calendar
        # C4: datetime.utcfromtimestamp() deprecated в Py3.12
        dt = datetime.fromtimestamp(calendar.timegm(dt), tz=timezone.utc)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    try:
        local = dt.astimezone()  # система определяет local timezone
    except Exception:
        local = dt  # fallback: UTC

    return local.strftime("%d.%m.%Y %H:%M")


from .console import ConsoleDelivery, Delivery
from .telegram import TelegramDelivery

__all__ = ["Delivery", "ConsoleDelivery", "TelegramDelivery", "format_time_utc"]