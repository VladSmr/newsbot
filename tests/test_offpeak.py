"""
Тесты off-peak гейта: is_deepseek_peak / is_llm_allowed_now.
Запуск: python -m pytest tests -q
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from newsbot.offpeak import is_deepseek_peak, is_llm_allowed_now


# ─────────────────────────────────────────────────────────────────
# Вспомогательные — dt(weekday, hour, min=0)
# weekday Python: 0=Пн, 1=Вт, 2=Ср, 3=Чт, 4=Пт, 5=Сб, 6=Вс
# ─────────────────────────────────────────────────────────────────

def dt(weekday: int, hour: int, minute: int = 0) -> datetime:
    """UTC datetime для заданного weekday (0=Пн) и часа, считая от 2026-01-05 (Пн)."""
    from datetime import timedelta
    base = datetime(2026, 1, 5, hour, minute, 0, tzinfo=timezone.utc)
    # weekday() 2026-01-05 = 0 (понедельник)
    delta = weekday - base.weekday()  # смещение в днях
    return base + timedelta(days=delta)


class TestIsDeepseekPeak:
    # ── пиковые часы (Пн–Пт, 01:00–03:59 и 06:00–09:59 UTC) ──

    def test_peak_monday_02_00(self):
        """Пн 02:00 UTC — пик (начало утреннего окна)."""
        assert is_deepseek_peak(dt(0, 2, 0)) is True

    def test_peak_monday_03_59(self):
        """Пн 03:59 UTC — пик (конец утреннего окна, полуоткрытый)."""
        assert is_deepseek_peak(dt(0, 3, 59)) is True

    def test_peak_monday_07_00(self):
        """Пн 07:00 UTC — пик (середина вечернего окна)."""
        assert is_deepseek_peak(dt(0, 7, 0)) is True

    def test_peak_monday_09_59(self):
        """Пн 09:59 UTC — пик (конец вечернего окна, полуоткрытый)."""
        assert is_deepseek_peak(dt(0, 9, 59)) is True

    def test_peak_tuesday_06_00(self):
        """Вт 06:00 UTC — пик (начало вечернего окна)."""
        assert is_deepseek_peak(dt(1, 6, 0)) is True

    def test_peak_thursday_01_00(self):
        """Чт 01:00 UTC — пик (начало утреннего окна)."""
        assert is_deepseek_peak(dt(3, 1, 0)) is True

    # ── границы — off-peak (полуоткрытые интервалы) ──

    def test_offpeak_04_00_boundary(self):
        """Пн 04:00 UTC — off-peak (04:00 уже за пределами утреннего пика)."""
        assert is_deepseek_peak(dt(0, 4, 0)) is False

    def test_offpeak_10_00_boundary(self):
        """Пн 10:00 UTC — off-peak (10:00 уже за пределами вечернего пика)."""
        assert is_deepseek_peak(dt(0, 10, 0)) is False

    def test_offpeak_01_00_boundary_is_peak(self):
        """Пн 01:00 UTC — пик (01:00 ∈ [01:00, 04:00) )."""
        assert is_deepseek_peak(dt(0, 1, 0)) is True

    def test_offpeak_00_59_boundary(self):
        """Пн 00:59 UTC — off-peak (00:59 < 01:00)."""
        assert is_deepseek_peak(dt(0, 0, 59)) is False

    def test_offpeak_06_00_boundary_is_peak(self):
        """Пн 06:00 UTC — пик (06:00 ∈ [06:00, 10:00) )."""
        assert is_deepseek_peak(dt(0, 6, 0)) is True

    def test_offpeak_05_59_boundary(self):
        """Пн 05:59 UTC — off-peak (05:59 < 06:00)."""
        assert is_deepseek_peak(dt(0, 5, 59)) is False

    # ── off-peak в будни (за пределами пиковых окон) ──

    def test_offpeak_monday_00_30(self):
        """Пн 00:30 UTC — off-peak (между полуночью и началом пика)."""
        assert is_deepseek_peak(dt(0, 0, 30)) is False

    def test_offpeak_monday_13_00(self):
        """Пн 13:00 UTC — off-peak (между пиковыми окнами и концом дня)."""
        assert is_deepseek_peak(dt(0, 13, 0)) is False

    def test_offpeak_monday_23_59(self):
        """Пн 23:59 UTC — off-peak."""
        assert is_deepseek_peak(dt(0, 23, 59)) is False

    def test_offpeak_friday_12_00(self):
        """Пт 12:00 UTC — off-peak."""
        assert is_deepseek_peak(dt(4, 12, 0)) is False

    # ── off-peak в выходные ──

    def test_offpeak_saturday_peak_time(self):
        """Сб 02:00 UTC — off-peak, несмотря на «пиковый час»."""
        assert is_deepseek_peak(dt(5, 2, 0)) is False

    def test_offpeak_saturday_07_00(self):
        """Сб 07:00 UTC — off-peak."""
        assert is_deepseek_peak(dt(5, 7, 0)) is False

    def test_offpeak_sunday_at_peak_hours(self):
        """Вс 09:30 UTC — off-peak."""
        assert is_deepseek_peak(dt(6, 9, 30)) is False


class TestIsLlmAllowedNow:
    def test_off_peak_only_false_always_allowed(self):
        """off_peak_only=False → всегда можно вызывать, даже в пик."""
        peak_time = dt(0, 2, 0)  # пн 02:00 UTC — пик
        assert is_llm_allowed_now(off_peak_only=False, now_utc=peak_time) is True

    def test_off_peak_only_true_peak_blocked(self):
        """off_peak_only=True + пик → заблокировано."""
        peak_time = dt(0, 7, 0)  # пн 07:00 UTC — пик
        assert is_llm_allowed_now(off_peak_only=True, now_utc=peak_time) is False

    def test_off_peak_only_true_offpeak_allowed(self):
        """off_peak_only=True + off-peak → разрешено."""
        offpeak_time = dt(0, 13, 0)  # пн 13:00 UTC — off-peak
        assert is_llm_allowed_now(off_peak_only=True, now_utc=offpeak_time) is True

    def test_off_peak_only_true_weekend_allowed(self):
        """off_peak_only=True + выходные → разрешено."""
        saturday = dt(5, 7, 0)  # сб 07:00 — off-peak (выходной)
        assert is_llm_allowed_now(off_peak_only=True, now_utc=saturday) is True


class TestTimezoneNormalization:
    """
    is_deepseek_peak / is_llm_allowed_now: tz-aware datetime нормализуются к UTC.
    """

    def test_is_deepseek_peak_tzaware_local_msk(self):
        """tz-aware datetime в MSK (UTC+3) → нормализуется к UTC."""
        from datetime import timedelta
        # Фиксированный UTC+3 (pytz-независимо)
        msk_offset = timezone(timedelta(hours=3))
        # Пн 07:30 MSK (UTC+3) = Пн 04:30 UTC → off-peak (04:30 не в пике)
        msk_dt = datetime(2026, 1, 5, 7, 30, 0, tzinfo=msk_offset)  # 2026-01-05 = понедельник
        assert is_deepseek_peak(msk_dt) is False

    def test_is_deepseek_peak_tzaware_utc_is_peak(self):
        """Пн 01:30 UTC ∈ [01:00, 04:00) → пик."""
        assert is_deepseek_peak(dt(0, 1, 30)) is True

    def test_is_llm_allowed_tzaware_normalization(self):
        """is_llm_allowed_now: tz-aware локальное время нормализуется к UTC."""
        from datetime import timedelta
        # Фиксированный UTC+3
        msk_offset = timezone(timedelta(hours=3))
        # Пн 07:30 MSK (UTC+3) = Пн 04:30 UTC → off-peak
        # без нормализации .hour был бы 7 (пик!)
        msk_dt = datetime(2026, 1, 5, 7, 30, 0, tzinfo=msk_offset)  # понедельник
        result = is_llm_allowed_now(off_peak_only=True, now_utc=msk_dt)
        assert result is True, "07:30 MSK (04:30 UTC) должно быть off-peak"