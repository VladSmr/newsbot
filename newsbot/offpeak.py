"""
Off-peak гейт для DeepSeek LLM.

Пиковые часы DeepSeek (UTC, только Пн–Пт):
  — 01:00–03:59 UTC (час 1, 2, 3)
  — 06:00–09:59 UTC (час 6, 7, 8, 9)
Всё остальное — off-peak (включая выходные).

NOTE: Китайские государственные праздники НЕ учитываются —
код консервативно считает будни пиковыми часами, т.е. модель будет
отключена в праздничный день (пик важнее экономии).

Границы полуоткрытые: 04:00 UTC уже off-peak, 10:00 UTC уже off-peak,
01:00 UTC — пик, 00:59 UTC — off-peak.

Moscow Time (UTC+3):
  — пик: 04:00–06:59 и 09:00–12:59 MSK, Пн–Пт
"""

from datetime import datetime, timezone
from typing import Optional


def is_deepseek_peak(now_utc: Optional[datetime] = None) -> bool:
    """
    Возвращает True, если текущее время попадает в пиковые часы DeepSeek.

    Пик: Пн–Пт (weekday 0..4) И (01:00–03:59 UTC ИЛИ 06:00–09:59 UTC).
    Границы полуоткрытые: 04:00 и 10:00 UTC — off-peak.
    Китайские праздники НЕ учитываются (см. модульный docstring).

    Args:
        now_utc: datetime с tzinfo. По умолчанию — datetime.now(timezone.utc).
                 Параметр для тестируемости.

    Returns:
        True — сейчас пиковое время (LLM вызывать НЕ рекомендуется).
        False — off-peak (LLM можно вызывать).
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    else:
        # защитно нормализуем к UTC: если объект tz-aware,
        # astimezone(timezone.utc) даёт корректный час независимо от локальной tz.
        now_utc = now_utc.astimezone(timezone.utc)

    # weekday: 0=Пн, 1=Вт, ..., 4=Пт, 5=Сб, 6=Вс
    if now_utc.weekday() > 4:
        return False  # выходной — off-peak

    hour = now_utc.hour

    # 01:00–03:59 UTC  (час 1, 2, 3)
    is_peak_morning = 1 <= hour < 4
    # 06:00–09:59 UTC  (час 6, 7, 8, 9)
    is_peak_evening = 6 <= hour < 10

    return is_peak_morning or is_peak_evening


def is_llm_allowed_now(off_peak_only: bool, now_utc: Optional[datetime] = None) -> bool:
    """
    Возвращает True, если LLM-вызов сейчас допустим.

    Args:
        off_peak_only: флаг из конфига llm.off_peak_only.
                       False → LLM всегда можно вызывать (текущее поведение).
                       True  → только в off-peak часы.
        now_utc: datetime с tzinfo. По умолчанию — datetime.now(timezone.utc).

    Returns:
        True — вызывать LLM можно.
        False — вызов следует пропустить (пик DeepSeek, off_peak_only=True).
    """
    if not off_peak_only:
        return True
    if now_utc is not None:
        # единообразная нормализация с is_deepseek_peak
        now_utc = now_utc.astimezone(timezone.utc)
    return not is_deepseek_peak(now_utc=now_utc)