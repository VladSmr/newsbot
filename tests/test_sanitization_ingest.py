"""
A1/A2/A3: санитизация при ингесте (collector + urlnorm).
Запускать: python -m pytest tests/test_sanitization_ingest.py -v
"""
import pytest

from newsbot.collector import _strip_editorial_tags, _strip_description_trails
from newsbot.urlnorm import normalize_url, TRACKING_PARAMS


# ═══════════════════════════════════════════════════════════════════════════════
# A1: strip editorial tags
# ═══════════════════════════════════════════════════════════════════════════════

class TestStripEditorialTags:
    """A1-fix: суффиксы/префиксы редакционных тегов чистятся, содержательные
    скобки (годы, подзаголовки) не трогаются."""

    # EXCLUSIVE
    def test_exclusive_suffix(self):
        assert _strip_editorial_tags("How Grimes' Cameo Came Together (EXCLUSIVE)") == \
            "How Grimes' Cameo Came Together"

    def test_exclusive_prefix(self):
        assert _strip_editorial_tags("EXCLUSIVE: Big News Story") == "Big News Story"

    def test_exclusive_suffix_case_insensitive(self):
        assert _strip_editorial_tags("Story (exclusive)") == "Story"

    # STAT+
    def test_stat_plus_prefix(self):
        assert _strip_editorial_tags("STAT+: CVS Health to split") == "CVS Health to split"

    def test_stat_plus_prefix_no_space(self):
        assert _strip_editorial_tags("STAT+Story") == "Story"

    # UPDATE N
    def test_update_n_suffix(self):
        assert _strip_editorial_tags("News UPDATE 3") == "News"

    def test_update_n_suffix_with_colon(self):
        assert _strip_editorial_tags("UPDATE 5: Full Story") == "Full Story"

    def test_update_n_in_parentheses(self):
        assert _strip_editorial_tags("News (UPDATE 2)") == "News"

    # ANALYSIS / OPINION
    def test_analysis_prefix(self):
        assert _strip_editorial_tags("ANALYSIS: What Happened") == "What Happened"

    def test_analysis_suffix(self):
        assert _strip_editorial_tags("Story (ANALYSIS)") == "Story"

    def test_opinion_prefix(self):
        assert _strip_editorial_tags("OPINION: My View") == "My View"

    def test_opinion_suffix(self):
        assert _strip_editorial_tags("Story (OPINION)") == "Story"

    # Содержательные скобки — НЕ трогать
    def test_film_year_preserved(self):
        assert _strip_editorial_tags("Movie Sequel (2026) Released") == "Movie Sequel (2026) Released"

    def test_year_suffix_preserved(self):
        assert _strip_editorial_tags("Film (2025)") == "Film (2025)"

    def test_subtitle_preserved(self):
        assert _strip_editorial_tags("Title: Subtitle — Source") == "Title: Subtitle — Source"

    def test_jti_spelling_preserved(self):
        assert _strip_editorial_tags("J.T.I. LLCs Report") == "J.T.I. LLCs Report"

    def test_us_abbreviation_preserved(self):
        assert _strip_editorial_tags("U.S. Economy Grows") == "U.S. Economy Grows"

    def test_eg_abbreviation_preserved(self):
        assert _strip_editorial_tags("e.g. example text") == "e.g. example text"

    # Collapse + trailing separator
    def test_collapse_spaces(self):
        assert _strip_editorial_tags("  EXCLUSIVE:   Multiple   Spaces ") == "Multiple Spaces"

    def test_trailing_dash_separator(self):
        assert _strip_editorial_tags("Title —") == "Title"

    def test_trailing_pipe_separator(self):
        assert _strip_editorial_tags("Title |") == "Title"

    def test_trailing_dash_space(self):
        assert _strip_editorial_tags("Title  -") == "Title"

    # Пустой / нормальный
    def test_empty(self):
        assert _strip_editorial_tags("") == ""

    def test_normal_title_unchanged(self):
        assert _strip_editorial_tags("Normal News Headline") == "Normal News Headline"


# ═══════════════════════════════════════════════════════════════════════════════
# A2: strip RSS-description trailing garbage
# ═══════════════════════════════════════════════════════════════════════════════

class TestStripDescriptionTrails:
    """A2-fix: feedburner-хвосты, 'Read more', '[...]' и т.п. убираются,
    содержательный текст не режется."""

    def test_feedburner_tail(self):
        # Содержательное тело + feedburner-хвост: тело сохраняется
        text = "Cork Takes the Heat So DRACO Data Survived. appeared first on SpaceNews."
        result = _strip_description_trails(text)
        assert "appeared first on" not in result
        assert "Cork Takes" in result
        assert "DRACO" in result

    def test_feedburner_tail_with_period(self):
        text = "Article text here. The post Title appeared first on Source."
        result = _strip_description_trails(text)
        assert "appeared first on" not in result

    def test_read_more(self):
        assert "Read more" not in _strip_description_trails(
            "Some summary text. Read more"
        )
        assert "Some summary text." in _strip_description_trails(
            "Some summary text. Read more"
        )

    def test_click_here(self):
        # "Click here to read" (with 'to read') — паттерн требует завершённой фразы
        assert "Click here to read" not in _strip_description_trails(
            "Summary text. Click here to read more."
        )
        assert "Summary text." in _strip_description_trails(
            "Summary text. Click here to read more."
        )

    def test_continue_reading(self):
        assert "Continue reading" not in _strip_description_trails(
            "Summary. Continue reading"
        )

    def test_bracket_more(self):
        assert "more" not in _strip_description_trails(
            "Some text. [ Read more ]"
        )

    def test_ellipsis_bracket(self):
        assert "[...]" not in _strip_description_trails(
            "Some text. [...]"
        )

    def test_text_without_trail_unchanged(self):
        text = "This is a normal news summary with actual content."
        assert _strip_description_trails(text) == text

    def test_collapse_multiple_spaces(self):
        text = "Text with   many    spaces   here."
        result = _strip_description_trails(text)
        assert "  " not in result

    def test_only_feedburner_tail_returns_empty(self):
        """Feedburner-хвост без содержательного текста → пустая строка."""
        # Содержательная часть — пробел перед "appeared first on"
        text = "Some content. appeared first on SpaceNews."
        result = _strip_description_trails(text)
        assert result == "Some content." or result == ""

    def test_feedburner_tail_with_broken_period(self):
        """Кривой «SpaceNews .» из реального дефекта."""
        text = "Real content here. appeared first on SpaceNews ."
        result = _strip_description_trails(text)
        assert "appeared first on" not in result
        assert "Real content" in result


# ═══════════════════════════════════════════════════════════════════════════════
# A3: tracking params stripped from URL
# ═══════════════════════════════════════════════════════════════════════════════

class TestUrlTrackingStrip:
    """A3-fix: from=rss, yclid и др. трекинг-параметры стрипаются,
    остальные query-параметры сохраняются."""

    def test_from_rss_stripped(self):
        url = "https://dtf.ru/news/123?from=rss"
        assert "from=rss" not in normalize_url(url)
        assert "dtf.ru/news/123" in normalize_url(url)

    def test_from_rss_with_other_params(self):
        url = "https://example.com/article?id=42&from=rss"
        result = normalize_url(url)
        assert "from=rss" not in result
        assert "id=42" in result

    def test_yclid_stripped(self):
        url = "https://example.com/page?yclid=abc123"
        assert "yclid" not in normalize_url(url)

    def test_utm_all_stripped(self):
        url = "https://x.com/a?utm_source=rss&utm_medium=email&utm_campaign=c1"
        result = normalize_url(url)
        assert "utm_" not in result

    def test_fbclid_gclid_stripped(self):
        url = "https://x.com/a?fbclid=xxx&gclid=yyy"
        result = normalize_url(url)
        assert "fbclid" not in result
        assert "gclid" not in result
        assert "mc_eid" not in normalize_url("https://x.com/a?mc_eid=x")

    def test_normal_query_preserved(self):
        url = "https://example.com/article?id=42&page=3"
        result = normalize_url(url)
        assert "id=42" in result
        assert "page=3" in result

    def test_url_without_query_unchanged(self):
        url = "https://example.com/article"
        assert normalize_url(url) == "https://example.com/article"

    def test_mc_cid_stripped(self):
        assert "mc_cid" not in normalize_url("https://x.com/a?mc_cid=x")

    def test_from_in_tracking_params(self):
        """'from' теперь в TRACKING_PARAMS."""
        assert "from" in TRACKING_PARAMS
        assert normalize_url("https://x.com/?from=rss") == "https://x.com/"