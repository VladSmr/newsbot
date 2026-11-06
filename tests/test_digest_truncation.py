"""
A4: обрезка summary по границе слова/предложения в дайджесте.
Запускать: python -m pytest tests/test_digest_truncation.py -v
"""
import pytest
import re

from newsbot.digest import _truncate_raw_summary


class TestTruncateAtSentenceBoundary:
    """A4-fix: обрезка по последнему предложению (". ") или слову,
    не посередине слова. Результат заканчивается «…»."""

    def test_ends_with_ellipsis(self):
        long_text = "A" * 600
        result = _truncate_raw_summary(long_text, 500)
        assert result.endswith("…")

    def test_no_ellipsis_for_short(self):
        short = "Short text."
        assert _truncate_raw_summary(short, 500) == short

    def test_exact_length_no_truncation(self):
        text = "A" * 300
        assert _truncate_raw_summary(text, 300) == text

    def test_at_sentence_boundary(self):
        """Обрезка на границе предложения: после точки + пробела."""
        text = "First sentence. Second sentence here. Third sentence final."
        # max_chars=40: before_limit ends "...Third s" — second ". " at pos 35 inside window
        result = _truncate_raw_summary(text, 40)
        assert result.endswith("…")
        assert result.rstrip("…").endswith("sentence here.")
        assert "Third" not in result

    def test_at_word_boundary(self):
        """Обрезка посреди текста без точки → на границе слова."""
        text = "First sentence with many words here and there."
        result = _truncate_raw_summary(text, 30)
        assert result.endswith("…")
        words = result.rstrip("…").split()
        assert len(words) > 0
        # Последнее слово целое
        assert len(words[-1]) <= 10

    def test_period_priority_over_space(self):
        """Точка с пробелом — приоритет над границей слова."""
        text = "Words words words word word. More text here."
        # full text = 46 chars; use 35 to truncate but still include ". " of "word."
        # rfind(". ", 0, 35) finds pos 21 (period of "word."), prefix=21 ≥ 4 → cut
        result = _truncate_raw_summary(text, 35)
        assert result.endswith("…")
        assert "More" not in result
        assert result.rstrip("…").endswith("word.")

    def test_mid_word_no_truncation(self):
        """Никогда не обрезать слово посередине."""
        text = "This is a test word is good."
        result = _truncate_raw_summary(text, 20)
        assert result.endswith("…")
        # Слово перед "…" должно быть целым (не обрезано посередине)
        words = result.rstrip("…").split()
        last = words[-1] if words else ""
        # "word" — целое слово, не "wor" или "d"
        assert last in ("test", "word", "is", "a", "This", "good"), \
            f"last_word={last!r}, result={result!r}"
        # Не обрезано посредине ("wor" не целое слово, "d" не слово)
        assert last not in ("wor", "d", "goo", "goo", "is ", "a ")

    def test_us_abbreviation_not_cut(self):
        """U.S. — точка в середине, не конец предложения. Обрезка после 'growing.'."""
        text = "The U.S. economy is growing. More details soon."
        result = _truncate_raw_summary(text, 30)
        assert result.endswith("…")
        # U.S. сохраняется; текст обрезан после "growing." (реальное предложение)
        assert "U.S." in result
        # Не обрезано на "U.S." — результат содержит текст после аббревиатуры
        assert re.search(r"growing", result), f"Expected 'growing' after U.S., got: {result!r}"

    def test_dr_abbreviation_not_cut(self):
        """Dr. — не конец предложения. Обрезка после 'Tuesday.' (реальное предложение)."""
        text = "Dr. Smith said the deal closed on Tuesday. More to come."
        # max=48: before_limit="Dr. Smith said the deal closed on Tuesday. More t"
        # — period after "Tuesday" at pos 37 is inside limit.
        # Abbreviation check: prefix "Dr. Smith said the deal closed on Tuesday" (37 chars)
        # has last-word "Tuesday"≥4, not trailing abbrev → cut after "Tuesday."
        result = _truncate_raw_summary(text, 48)
        assert result.endswith("…")
        assert "Dr." in result
        # Обрезано после "Tuesday." — реальное предложение, не "Dr."
        assert "Tuesday." in result, f"Expected 'Tuesday.' in result, got: {result!r}"

    def test_eg_abbreviation_not_sentence_end(self):
        """e.g. — не конец предложения. Обрезка после 'happened.' (реальное предложение)."""
        text = "e.g. this happened. Now the full story begins."
        result = _truncate_raw_summary(text, 25)
        assert result.endswith("…")
        # Обрезано после "happened." — e.g. пропущено, далее идёт "Now the full..."
        assert "happened." in result, f"Expected 'happened.' in result, got: {result!r}"
        assert "e.g." in result  # аббревиатура сохранена

    def test_multi_sentence_cut_at_last_period(self):
        """Три предложения — обрезка после последней точки перед лимитом."""
        text = "First. Second sentence here. Third and final."
        # max_chars=36: before_limit ends "...Second sentence here." —
        # second ". " at pos 29 inside window → cut at "here."
        result = _truncate_raw_summary(text, 36)
        assert result.endswith("…")
        assert "Third" not in result
        assert result.rstrip("…").endswith("here.")

    def test_no_periods_cut_at_space(self):
        """Текст без точек — только граница слова."""
        text = "A long text without periods and many words"
        result = _truncate_raw_summary(text, 20)
        assert result.endswith("…")
        words = result.rstrip("…").split()
        assert len(words[-1]) < len(text)
        # Не заканчивается на "words" (слово не обрезано посередине)
        assert not result.rstrip("…").endswith("words")

    def test_period_without_space_not_sentence_end(self):
        """Точка без пробела (число, аббревиатура) — не конец предложения."""
        text = "Item 1. Item 2. Item 3 now available."
        result = _truncate_raw_summary(text, 20)
        assert result.endswith("…")
        words = result.rstrip("…").split()
        # "1." и "2." не sentence-end; граница слова
        assert all(len(w) > 0 for w in words)
        # Не обрезано посередине слова
        assert not result.rstrip("…").endswith("ble")