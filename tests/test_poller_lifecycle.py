"""
Тесты арифметики слотов демона: расчёт секунд до следующего слота и форматирование часа слота.

Запускать: python -m pytest tests/test_poller_lifecycle.py -v
"""
from datetime import datetime as real_dt_cls


class TestNextSlotSeconds:
    """Расчёт секунд до ближайшего слота :30 (чётный час, local time)."""

    def test_10_05_returns_25_min(self):
        from newsbot.__main__ import _next_slot_seconds
        now = real_dt_cls(2000, 1, 1, 10, 5, 0)
        assert _next_slot_seconds(now) == 25 * 60

    def test_09_05_odd_hour_returns_85_min(self):
        from newsbot.__main__ import _next_slot_seconds
        now = real_dt_cls(2000, 1, 1, 9, 5, 0)
        assert _next_slot_seconds(now) == 85 * 60

    def test_10_31_returns_119_min(self):
        """10:31 (>=:30) → слот 12:30, 119 мин."""
        from newsbot.__main__ import _next_slot_seconds
        now = real_dt_cls(2000, 1, 1, 10, 31, 0)
        assert _next_slot_seconds(now) == 119 * 60

    def test_10_30_returns_120_min(self):
        from newsbot.__main__ import _next_slot_seconds
        now = real_dt_cls(2000, 1, 1, 10, 30, 0)
        assert _next_slot_seconds(now) == 120 * 60

    def test_23_59_returns_31_min_next_day(self):
        from newsbot.__main__ import _next_slot_seconds
        now = real_dt_cls(2000, 1, 1, 23, 59, 0)
        assert _next_slot_seconds(now) == 31 * 60

    def test_00_29_returns_1_min(self):
        from newsbot.__main__ import _next_slot_seconds
        now = real_dt_cls(2000, 1, 1, 0, 29, 0)
        assert _next_slot_seconds(now) == 60

    def test_00_30_returns_120_min(self):
        from newsbot.__main__ import _next_slot_seconds
        now = real_dt_cls(2000, 1, 1, 0, 30, 0)
        assert _next_slot_seconds(now) == 120 * 60

    def test_00_31_returns_119_min(self):
        """00:31 (>=:30) → слот 02:30, 119 мин."""
        from newsbot.__main__ import _next_slot_seconds
        now = real_dt_cls(2000, 1, 1, 0, 31, 0)
        assert _next_slot_seconds(now) == 119 * 60

    def test_slot_always_on_even_hour(self):
        """Каждый now даёт слот на чётный час, и всегда в будущее (>0)."""
        from newsbot.__main__ import _next_slot_seconds, _format_slot_hour
        test_cases = [
            (real_dt_cls(2000, 1, 1, 10, 5),  10),
            (real_dt_cls(2000, 1, 1, 10, 31), 12),
            (real_dt_cls(2000, 1, 1, 11, 5),  12),
            (real_dt_cls(2000, 1, 1, 12, 5),  12),
            (real_dt_cls(2000, 1, 1, 12, 31), 14),
            (real_dt_cls(2000, 1, 1, 13, 0),  14),
            (real_dt_cls(2000, 1, 1, 22, 29), 22),
            (real_dt_cls(2000, 1, 1, 22, 31), 0),
            (real_dt_cls(2000, 1, 1, 23, 59), 0),
            (real_dt_cls(2000, 1, 1, 0, 30),  2),
            (real_dt_cls(2000, 1, 1, 1, 0),   2),
            (real_dt_cls(2000, 1, 1, 1, 31),  2),
            (real_dt_cls(2000, 1, 1, 3, 0),   4),
        ]
        for now, expected_hour in test_cases:
            slot_hour = _format_slot_hour(now)
            assert slot_hour == expected_hour, (
                f"now={now.strftime('%H:%M')} → slot hour={slot_hour}, expected {expected_hour}"
            )
            result = _next_slot_seconds(now)
            assert result > 0, f"now={now.strftime('%H:%M')} → {result}s (must be > 0)"

    def test_10_35_returns_115_min(self):
        """10:35 (>=:30) → слот 12:30, 115 мин. Точный ассерт (не >0)."""
        from newsbot.__main__ import _next_slot_seconds
        secs = _next_slot_seconds(real_dt_cls(2000, 1, 1, 10, 35, 0))
        assert secs == 115 * 60


class TestNextWaitSecondsAfterError:
    """
    MAJOR-фикс: error-путь подключён к реальному циклу, пауза считается от t0.
    `_next_wait_seconds(t0, after_error=True) == max(backoff, slot_from_t0)`.
    """

    def test_after_error_false_equals_slot(self):
        """after_error=False → ровно слот, без backoff."""
        from newsbot.__main__ import _next_wait_seconds
        now = real_dt_cls(2000, 1, 1, 10, 5, 0)
        assert _next_wait_seconds(now, after_error=False) == 25 * 60

    def test_after_error_slot_greater_than_backoff(self):
        """slot > backoff → after_error=True даёт ровно slot (1500 с, не 30 с)."""
        from newsbot.__main__ import _next_wait_seconds
        now = real_dt_cls(2000, 1, 1, 10, 5, 0)  # slot = 25 мин = 1500 с > 30
        assert _next_wait_seconds(now, after_error=True) == 25 * 60

    def test_after_error_slot_less_than_backoff(self):
        """slot < backoff → after_error=True даёт ровно backoff (30 с, не slot=10 с).

        Сценарий: ошибка в 10:29:50 (до слота 10:30 осталось 10 с).
        Пауза = max(backoff=30, slot=10) = 30 с.
        Текущий цикл досыпает 30 с → слот 10:30 НЕ теряется.
        """
        from newsbot.__main__ import _next_wait_seconds, _BACKOFF_SECONDS
        now = real_dt_cls(2000, 1, 1, 10, 29, 50)  # slot = 10 с < backoff=30 с
        assert _next_wait_seconds(now, after_error=True) == _BACKOFF_SECONDS

    def test_after_error_slot_equal_to_backoff(self):
        """slot == backoff → after_error=True даёт ровно это значение."""
        from newsbot.__main__ import _next_wait_seconds, _BACKOFF_SECONDS
        # Найти момент, когда slot_secs == _BACKOFF_SECONDS == 30.
        # 00:29 → 60 с (не равно). Проверяем через after_error=False.
        from newsbot.__main__ import _next_slot_seconds
        # Подбираем now так, чтобы slot_secs == 30 (т.е. до слота ровно 30 с).
        # Slot :30 на чётный час. Если now = HH:29:30 → slot = 30 с.
        now = real_dt_cls(2000, 1, 1, 10, 29, 30)
        assert _next_slot_seconds(now) == 30
        assert _next_wait_seconds(now, after_error=True) == 30

    def test_after_error_t0_preserves_slot(self):
        """Пауза считается от t0 (момента ошибки), не от now после backoff.

        Сценарий из описания MAJOR-фикса:
        - Ошибка в 10:29:50 → t0 = 10:29:50 → slot_from_t0 = 10 с.
        - После backoff (30 с) now ≈ 10:30:20 → slot_from_now ≈ 120 мин.
        - after_error=True → max(30, 10) = 30 с (слот 10:30 сохраняется).
        - БЕЗ after_error → пауза была бы ≈120 мин — слот 10:30 теряется.

        БЕЗ реального sleep, БЕЗ threading.Event, БЕЗ вызова main().
        """
        from newsbot.__main__ import _next_wait_seconds, _next_slot_seconds, _BACKOFF_SECONDS

        t0 = real_dt_cls(2000, 1, 1, 10, 29, 50)   # момент ошибки
        after_now = real_dt_cls(2000, 1, 1, 10, 30, 20)  # now ПОСЛЕ backoff

        # Пауза считается от t0: max(backoff=30, slot_from_t0=10) = 30 с.
        wait_from_t0 = _next_wait_seconds(t0, after_error=True)
        assert wait_from_t0 == _BACKOFF_SECONDS, (
            f"slot от t0 потерян: ожидалось {_BACKOFF_SECONDS} с, получено {wait_from_t0} с"
        )

        # Пауза от after_now (без after_error) дала бы ≈120 мин — слот 10:30 потерян.
        slot_from_after_now = _next_slot_seconds(after_now)
        assert slot_from_after_now > _BACKOFF_SECONDS, (
            "sanity: slot_from_after_now должен быть > 30 с (иначе тест бессмыслен)"
        )


class TestResolveWait:
    """
    Тесты чистой функции _resolve_wait: выбор опорного времени и расчёт паузы.

    Регрессионные тесты на баг: t0 не обновлялся на успешном пути →
    пауза считалась от момента старта процесса, а не от текущего времени.
    """

    def test_success_path_uses_now_not_t0_error(self):
        """
        Регрессионная ловушка: если t0_error остался от старта (10:05),
        а now = 10:07, пауза должна быть 23 мин (до слота 10:30), а не 25 мин.

        10:07 → ближайший слот 10:30 = 23 мин = 1380 с.
        Если бы использовался t0_error=10:05 → слот 10:30 = 25 мин = 1500 с.
        """
        from newsbot.__main__ import _resolve_wait, _next_wait_seconds
        from datetime import datetime as dt_cls

        t0_error = dt_cls(2000, 1, 1, 10, 5, 0)   # момент старта процесса
        now = dt_cls(2000, 1, 1, 10, 7, 0)       # момент после успешного цикла

        # after_error=False → опора = now = 10:07
        wait = _resolve_wait(t0_error, after_error=False, now=now)
        assert wait == 23 * 60, (
            f"Успешный путь должен считать от now=10:07 → 23 мин (1380 с), "
            f"получено {wait} с. Если 1500 с — t0_error всё ещё используется!"
        )

    def test_success_path_explicit_now_examples(self):
        """after_error=False всегда использует now, независимо от t0_error."""
        from newsbot.__main__ import _resolve_wait
        from datetime import datetime as dt_cls

        # t0_error далёкий, now близко к слоту
        t0_error = dt_cls(2000, 1, 1, 8, 0, 0)
        now = dt_cls(2000, 1, 1, 10, 28, 0)  # до слота 10:30 = 2 мин
        wait = _resolve_wait(t0_error, after_error=False, now=now)
        assert wait == 2 * 60

    def test_error_path_uses_t0_error(self):
        """after_error=True использует t0_error (момент ошибки), не now."""
        from newsbot.__main__ import _resolve_wait, _BACKOFF_SECONDS
        from datetime import datetime as dt_cls

        t0_error = dt_cls(2000, 1, 1, 10, 29, 50)  # ошибка за 10 с до слота
        now = dt_cls(2000, 1, 1, 10, 30, 20)        # now после backoff

        wait = _resolve_wait(t0_error, after_error=True, now=now)
        assert wait == _BACKOFF_SECONDS, (
            f"Ошибка в 10:29:50 → max(backoff={_BACKOFF_SECONDS}, slot=10) = "
            f"{_BACKOFF_SECONDS} с. Получено {wait} с"
        )


class TestFormatSlotHour:
    """Форматирование часа слота для логирования."""

    def test_various_times(self):
        from newsbot.__main__ import _format_slot_hour
        cases = [
            (real_dt_cls(2000, 1, 1, 10, 5),  10),
            (real_dt_cls(2000, 1, 1, 10, 31), 12),
            (real_dt_cls(2000, 1, 1, 9, 31),  10),
            (real_dt_cls(2000, 1, 1, 23, 59), 0),
            (real_dt_cls(2000, 1, 1, 0, 29),  0),
            (real_dt_cls(2000, 1, 1, 0, 31),  2),
        ]
        for now, expected in cases:
            assert _format_slot_hour(now) == expected, (
                f"{now.strftime('%H:%M')} → {expected}"
            )