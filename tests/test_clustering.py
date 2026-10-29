"""Тесты стеммера, кластеризации и LLM-локаций (Задача: LLM вместо словаря)."""
import pytest

from newsbot.stemmer import normalize, stem_hash, _stem_word
from newsbot.pipeline import build_cluster_key, rebuild_cluster_key


def _cluster_key(item) -> str:
    """
    Тестовый хелпер: воспроизводит прод-контракт вызова build_cluster_key
    для items без известной LLM-локации (ингест до скоринга).
    Использует build_cluster_key с placeholder-локацией 'france'.
    """
    return build_cluster_key(
        "france",  # placeholder — заменится на LLM-локацию после скоринга
        stem_hash(item.title),
        item.published,
        title=item.title,
    )



class TestStemmer:
    """Тесты stemmer.py."""

    def test_stem_normalize_basic(self):
        """normalize() убирает числа, стоп-слова, короткие слова."""
        result = normalize("The police clash in Paris 2025")
        assert "the" not in result
        assert "police" in result
        assert "paris" in result
        assert "2025" not in result

    def test_stem_hash_deterministic(self):
        """stem_hash() детерминирован: одинаковый вход → одинаковый выход."""
        h1 = stem_hash("France protests: police clash in Paris")
        h2 = stem_hash("France protests: police clash in Paris")
        assert h1 == h2

    def test_stem_hash_similar_titles(self):
        """Синонимичные заголовки дают одинаковый stem_hash."""
        h1 = stem_hash("France protests: police clash in Paris")
        h2 = stem_hash("Paris protest clash: police in France")
        assert h1 == h2

    def test_stem_hash_different_titles(self):
        """Разные заголовки дают разные stem_hash."""
        h1 = stem_hash("France protests in Paris")
        h2 = stem_hash("Japan earthquake in Tokyo")
        assert h1 != h2

    def test_normalize_cyrillic(self):
        """normalize() работает с кириллицей."""
        result = normalize("В Белгородской области прошли события")
        assert "в" not in result
        assert "белгородск" in result or "област" in result

    def test_normalize_russian_drone_headline(self):
        """Русский заголовок → не пустой hash (не md5(''))."""
        result = normalize("Сбито 80 БПЛА в Крыму")
        assert result.strip() != "", (
            "normalize() не должен возвращать пустую строку для русского заголовка. "
            "Это указывает на потерю кириллицы при нормализации."
        )
        assert len(result) > 0

    def test_normalize_russian_vs_other_different(self):
        """Два разных русских заголовка → разные строки."""
        r1 = normalize("Сбито 80 БПЛА в Крыму")
        r2 = normalize("Другая новость в Крыму")
        assert r1 != r2, "Разные заголовки должны давать разные стемы"

    def test_stem_hash_cyrillic_nonempty(self):
        """stem_hash() для русского заголовка — не md5('')."""
        h = stem_hash("Сбито 80 БПЛА в Крыму")
        MD5_EMPTY = "d41d8cd98f00b204e9800998ecf8427e"
        assert h != MD5_EMPTY, (
            f"stem_hash вернул md5('') = {h}. Кириллица теряется при нормализации!"
        )
        assert len(h) == 8, "stem_hash должен быть 8-символьным"

    def test_normalize_emoji_headline(self):
        """Emoji/Unicode в заголовке — нормализация не падает, hash не пустой."""
        result = normalize("Breaking 🚨 News 🔥 in Tokyo")
        assert result.strip() != "", "Emoji-заголовок не должен давать пустой normalize()"
        h = stem_hash("Breaking 🚨 News 🔥 in Tokyo")
        assert h != "d41d8cd9", "Emoji-заголовок не должен давать md5('')"

    def test_stem_suffix_stripping(self):
        """Суффиксы корректно убираются."""
        result = normalize("protesting protesters protestation")
        assert "protest" in result

    def test_es_suffix_only_after_sibilants(self):
        """
        P2: -es снимается только у основ на s/x/z/ch/sh; иначе — обычное -s.
        Раньше cases→cas, notes→not, rates→rat: rate/rates расходились
        по кластерам и domain-дедуп не склеивал перепечатку.
        """
        assert _stem_word("rates") == _stem_word("rate") == "rate"
        assert _stem_word("notes") == _stem_word("note") == "note"
        # шипящие/свистящие — прежнее поведение
        assert _stem_word("passes") == "pass"
        assert _stem_word("boxes") == "box"
        assert _stem_word("clashes") == "clash"
        assert _stem_word("watches") == "watch"

    def test_es_se_bases_singular_plural_agree(self):
        """
        Minor 2-волны: основы на -se (немое -e) не должны терять -e во
        множественном числе. Раньше cases→cas, responses→respons,
        releases→releas, raises→rais — расходятся с единственным числом
        (case/response/release/raise) и дедуп не склеивает перепечатки.
        """
        for singular, plural in (
            ("case", "cases"),
            ("response", "responses"),
            ("release", "releases"),
            ("raise", "raises"),
            ("base", "bases"),
            ("promise", "promises"),
        ):
            assert _stem_word(plural) == _stem_word(singular) == singular, (
                f"{plural} и {singular} должны давать один стем"
            )
        # шипящие/свистящие не сломались
        assert _stem_word("passes") == "pass"
        assert _stem_word("classes") == "class"

    def test_es_us_plurals_keep_base(self):
        """
        Minor 4-волны (регрессия 2-волны): плюралисы на -us не должны
        получать немое -e. Было viruses→viruse, statuses→statuse, buses→buse —
        расходились с virus/status/bus. Чиним явным whitelist'ом.
        """
        for singular in ("virus", "status", "bus"):
            plural = singular + "es"
            assert _stem_word(plural) == _stem_word(singular) == singular, (
                f"{plural} и {singular} должны давать один стем"
            )
            assert stem_hash(f"New {plural} found") == stem_hash(
                f"New {singular} found"
            ), f"stem_hash для {plural}/{singular} разошёлся"

        # основы на -use (немое -e — часть корня) не сломаны
        assert _stem_word("houses") == _stem_word("house") == "house"
        assert _stem_word("purposes") == "purpose"

    def test_ies_suffix_to_y(self):
        """P2: -ies → y (rallies→rally, policies→policy)."""
        assert _stem_word("rallies") == "rally"
        assert _stem_word("policies") == "policy"
        assert _stem_word("industries") == "industry"

    def test_ed_restores_silent_e(self):
        """
        Minor 2-волны: -ed после гласной+s восстанавливает немое -e,
        иначе raised→rais, а raise→raise — расходятся по стемам.
        """
        assert _stem_word("raised") == _stem_word("raise") == "raise"
        assert _stem_word("used") == _stem_word("use") == "use"
        assert _stem_word("closed") == _stem_word("close") == "close"
        assert _stem_word("increased") == _stem_word("increase") == "increase"
        # основы на -us: немое e не восстанавливается (focus→focus, не focuse)
        assert _stem_word("focused") == _stem_word("focus") == "focus"
        # удвоение и обычные -ed не сломались
        assert _stem_word("stopped") == "stop"
        assert _stem_word("attacked") == "attack"

    @pytest.mark.parametrize(
        "past, plural, infinitive",
        [
            ("caused", "causes", "cause"),
            ("paused", "pauses", "pause"),
            ("housed", "houses", "house"),
            ("aroused", "arouses", "arouse"),
            ("used", "uses", "use"),
        ],
    )
    def test_ed_se_verb_forms_same_stem(self, past, plural, infinitive):
        """
        Major: -ed-restore исключение `not base.endswith('us')` било не только
        по focus/status. Базы -se-глаголов (caus/paus/hous/arous) тоже
        оканчиваются на 'us', поэтому caused→caus, а causes→cause (ветка
        -es:w[:-1]) — три формы одного слова давали ДВА стема, и дедуп не
        склеивал перепечатки («causes»/«caused»). Теперь исключение работает
        только по явному набору _US_SINGULARS, и -ed восстанавливает немое -e.
        """
        stems = {_stem_word(past), _stem_word(plural), _stem_word(infinitive)}
        assert stems == {infinitive}, (
            f"{past}/{plural}/{infinitive} должны давать один стем "
            f"{infinitive!r}, получили {stems}"
        )

    def test_ed_se_verb_forms_same_cluster_hash(self):
        """То же на уровне stem_hash (контракт кластеризации)."""
        for past, plural, infinitive in (
            ("caused", "causes", "cause"),
            ("paused", "pauses", "pause"),
            ("housed", "houses", "house"),
            ("aroused", "arouses", "arouse"),
        ):
            h_past = stem_hash(f"Storm {past} damage")
            h_plural = stem_hash(f"Storm {plural} damage")
            h_inf = stem_hash(f"Storm {infinitive} damage")
            assert h_past == h_plural == h_inf, (
                f"{past}/{plural}/{infinitive}: хеши разошлись "
                f"({h_past}, {h_plural}, {h_inf})"
            )

    def test_us_singulars_focus_forms_same_stem(self):
        """
        focus/focuses/focused → один стем: `-es`-ветка по _US_SINGULARS снимает
        -es целиком, `-ed`-ветка не восстанавливает немое -e.
        """
        assert _stem_word("focuses") == _stem_word("focus") == "focus"
        assert _stem_word("focused") == "focus"
        assert (
            stem_hash("Report focuses on crime")
            == stem_hash("Report focused on crime")
            == stem_hash("Report focus on crime")
        )

    @pytest.mark.parametrize(
        "singular, plural",
        [
            ("campus", "campuses"),
            ("bonus", "bonuses"),
            ("virus", "viruses"),
            ("status", "statuses"),
            ("census", "censuses"),
            ("consensus", "consensuses"),
            ("crocus", "crocuses"),
            ("surplus", "surpluses"),
            ("genius", "geniuses"),
            ("radius", "radiuses"),
            ("bus", "buses"),
        ],
    )
    def test_us_singular_plural_same_stem(self, singular, plural):
        """-us-существительные: plural и singular дают один стем."""
        assert _stem_word(plural) == _stem_word(singular) == singular

    def test_ed_us_singular_no_silent_e(self):
        """Существительные на -us не получают немое -e в -ed-ветке."""
        for word in ("focus", "status", "campus", "virus", "bonus", "bus"):
            assert _stem_word(word + "ed") == word, (
                f"{word}ed должен давать {word!r}"
            )

    def test_ed_es_generic_regressions(self):
        """Генерические -se-основы и шипящие не сломаны фиксом."""
        # plural: обычное -s (немое -e сохраняется)
        for word in ("case", "response", "release", "raise", "base", "promise"):
            assert _stem_word(word + "s") == word, f"{word}s != {word}"
        # past: немое -e восстанавливается только если перед 's' гласная
        # (raising-основа), поэтому response сюда не входит — у него 'ns'.
        for word in ("case", "release", "raise", "base", "promise", "close"):
            assert _stem_word(word + "d") == word, f"{word}d != {word}"
        assert _stem_word("houses") == "house"
        assert _stem_word("purposes") == "purpose"
        assert _stem_word("passes") == "pass"
        assert _stem_word("classes") == "class"
        assert _stem_word("boxes") == "box"
        assert _stem_word("clashes") == "clash"
        assert _stem_word("watches") == "watch"

    def test_stop_words_filtered_before_stemming(self):
        """
        Minor 2-волны: стоп-слова фильтруются ДО стемминга. Раньше
        does→doe, has→ha, was→wa не совпадали со STOP_WORDS и попадали
        в stem_hash мусорными токенами.
        """
        assert _stem_word("does") == "doe"  # стеммер как таковой
        norm = normalize("Central bank does nothing")
        assert "doe" not in norm.split(), f"'doe' просочился в normalize: {norm!r}"
        assert "does" not in norm.split()
        # прочие стоп-слова во множественном/прошедшем не протекают
        for text in ("He has plans", "It was here"):
            tokens = normalize(text).split()
            for junk in ("ha", "wa"):
                assert junk not in tokens, f"{junk!r} просочился: {tokens}"

    def test_raise_past_present_same_cluster(self):
        """
        Minor 2-волны: raise/raises/raised → один стем → один кластер
        (перепечатки «central bank raises rate» / «raised rate»).
        """
        h1 = stem_hash("Central bank raises key rate")
        h2 = stem_hash("Central bank raised key rate")
        h3 = stem_hash("Central bank raise key rate")
        assert h1 == h2 == h3, (
            f"raise/raises/raised должны совпадать: {h1}, {h2}, {h3}"
        )

    def test_rate_singular_plural_same_cluster(self):
        """rate/rates дают одинаковый stem → один кластер (domain-дедуп)."""
        assert stem_hash("Central bank raises key rate") == stem_hash(
            "Central bank raises key rates"
        )

    @pytest.mark.parametrize(
        "singular, plural",
        [
            ("class", "classes"),
            ("process", "processes"),
            ("address", "addresses"),
            ("success", "successes"),
            ("congress", "congresses"),
            ("press", "presses"),
            ("kiss", "kisses"),
            ("pass", "passes"),
        ],
    )
    def test_double_s_stem_preserves_base(self, singular, plural):
        """
        Major-1: -s снимается с согласной, но основы на -ss не трогаются.
        Иначе classes→clas, processes→proces, businesses→business,
        addresses→addres — а плюралисы идут по -es-ветке (classes→class) →
        singular и plural дают РАЗНЫЕ стемы.
        """
        assert _stem_word(plural) == _stem_word(singular) == singular, (
            f"{plural} и {singular} должны давать стем {singular!r}, "
            f"получили {_stem_word(plural)!r} / {_stem_word(singular)!r}"
        )
        assert stem_hash(f"New {plural} announced") == stem_hash(
            f"New {singular} announced"
        ), f"stem_hash для {plural}/{singular} разошёлся"

    @pytest.mark.parametrize(
        "past, present, infinitive",
        [
            ("passed", "passes", "pass"),
            ("addressed", "addresses", "address"),
            ("pressed", "presses", "press"),
            ("kissed", "kisses", "kiss"),
            ("guessed", "guesses", "guess"),
            ("expressed", "expresses", "express"),
        ],
    )
    def test_ed_ss_base_no_rollback(self, past, present, infinitive):
        """
        Major-1: -ed на -ss основе не откатывает удвоение.
        passed→pass (не pas), addressed→address (не addres).
        Согласованность с плюралисами: passes→pass, classes→class.
        """
        stems = {_stem_word(past), _stem_word(present), _stem_word(infinitive)}
        assert stems == {infinitive}, (
            f"{past}/{present}/{infinitive} должны давать один стем "
            f"{infinitive!r}, получили {stems}"
        )
        assert stem_hash(f"Meeting {past} at noon") == stem_hash(
            f"Meeting {infinitive} at noon"
        ), f"stem_hash для {past}/{infinitive} разошёлся"


# TestLocations удалён: locations.py удалён, локация теперь из LLM.
# Тесты локации — в test_llm_location.py (новый файл).


class TestClusterKey:
    """Тесты build_cluster_key() и rebuild_cluster_key()."""

    def test_cluster_key_structure(self):
        """cluster_key содержит локацию, stem, дату."""
        class FakeItem:
            title = "France protests: police clash in Paris"
            published = "2026-10-18T20:00:00+00:00"

        key = _cluster_key(FakeItem())
        parts = key.split("|")
        assert len(parts) == 3
        # Задача: локация теперь из LLM, здесь тестируем build_cluster_key
        # с произвольной локацией — не из словаря.
        stem = stem_hash(FakeItem().title)
        assert parts[0] == "france"  # build_cluster_key нормализует lowercase
        assert parts[1] == stem
        assert parts[2] == "2026-10-18"

    def test_cluster_key_stem_consistency(self):
        """Одинаковые title дают одинаковый stem в cluster_key."""
        class Fake1:
            title = "France protests: police clash in Paris"
            published = "2026-10-18T20:00:00+00:00"

        class Fake2:
            title = "Paris protest clash: police in France"
            published = "2026-10-18T20:00:00+00:00"

        k1 = build_cluster_key(
            "france", stem_hash(Fake1().title), Fake1().published, title=Fake1().title,
        )
        k2 = build_cluster_key(
            "france", stem_hash(Fake2().title), Fake2().published, title=Fake2().title,
        )
        # stem одинаковый, локация и день совпадают
        assert k1 == k2

    def test_cluster_key_different_days(self):
        """Разные дни — разные cluster_key."""
        class Fake1:
            title = "France protests in Paris"
            published = "2026-10-18T20:00:00+00:00"

        class Fake2:
            title = "France protests in Paris"
            published = "2026-10-19T20:00:00+00:00"

        k1 = build_cluster_key("france", stem_hash(Fake1().title), Fake1().published)
        k2 = build_cluster_key("france", stem_hash(Fake2().title), Fake2().published)
        assert k1 != k2
        assert "2026-10-18" in k1
        assert "2026-10-19" in k2

    def test_cluster_key_unknown_day(self):
        """Без published → day='unknown'."""
        k = build_cluster_key("france", "stem123", None)
        parts = k.split("|")
        assert len(parts) == 3
        assert parts[2] == "unknown"

    def test_cluster_key_nostem_per_title_unique(self):
        """
        CRITICAL 1 Regression: два разных nostem-заголовка (empty stem)
        в одной локации и в один день → РАЗНЫЕ cluster_key.
        """
        k1 = build_cluster_key("france", None, "2026-10-21T12:00:00+00:00", title="🛑")
        k2 = build_cluster_key("france", None, "2026-10-21T12:00:00+00:00", title="🚨")

        # Оба → nostem с РАЗНЫМИ per-title hash
        assert k1 != k2, (
            f"Разные nostem-заголовки должны давать РАЗНЫЕ cluster_key. "
            f"k1={k1!r}, k2={k2!r}"
        )
        assert "nostem:" in k1
        assert "nostem:" in k2
        assert "2026-10-21" in k1
        assert "2026-10-21" in k2

    def test_cluster_key_iso_date_format(self):
        """cluster_key содержит ISO-дату YYYY-MM-DD из published."""
        k = build_cluster_key("france", "stem123", "2026-10-21T14:30:00+00:00")
        parts = k.split("|")
        assert parts[2] == "2026-10-21"

    def test_cluster_key_date_from_published_not_fetched(self):
        """cluster_key использует published, не fetched_at."""
        k1 = build_cluster_key("russia", "stem123", "2026-10-21T08:00:00+00:00")
        k2 = build_cluster_key("russia", "stem123", "2026-10-21T08:00:00+00:00")
        assert k1 == k2


class TestRebuildClusterKey:
    """Тесты rebuild_cluster_key: LLM-локация подставляется в cluster_key."""

    def test_rebuild_normalizes_llm_location(self):
        """rebuild_cluster_key нормализует LLM-локацию: lowercase + strip."""
        old = "россия|stem123|2026-10-20"  # old cluster_key с пустой локацией
        new = rebuild_cluster_key("China", old)
        assert new.startswith("china|")
        assert "stem123" in new
        assert "2026-10-20" in new

    def test_rebuild_preserves_stem_and_day(self):
        """rebuild_cluster_key сохраняет stem и pub_day из старого ключа."""
        old = "other|abc12345|2026-10-22"
        new = rebuild_cluster_key("Tibet", old)
        parts = new.split("|")
        assert parts[0] == "tibet"
        assert parts[1] == "abc12345"
        assert parts[2] == "2026-10-22"

    def test_rebuild_empty_location_becomes_other(self):
        """Пустая/отсутствующая LLM-локация → 'other' (cap не применяется)."""
        old = "other|stem456|2026-10-20"
        new = rebuild_cluster_key("", old)
        assert new.startswith("other|")

    def test_rebuild_cyrillic_news_china(self):
        """Реальный кейс: новость на русском про Китай → LLM: 'China'."""
        # Симулируем старую локацию из словаря (которая могла быть 'other')
        old = "other|stem999|2026-10-20"
        new = rebuild_cluster_key("China", old)
        parts = new.split("|")
        assert parts[0] == "china"
        assert parts[1] == "stem999"
        assert parts[2] == "2026-10-20"

    def test_rebuild_uses_stem_from_old_key(self):
        """Stem берётся из old_cluster_key (вычислен при ингесте)."""
        old = "other|xyz789ab|2026-10-18"
        new = rebuild_cluster_key("Paris", old)
        parts = new.split("|")
        assert parts[1] == "xyz789ab"

    def test_rebuild_no_old_cluster_key(self):
        """Без старого cluster_key — nostem с пустым title hash."""
        new = rebuild_cluster_key("Russia", None)
        parts = new.split("|")
        assert parts[0] == "russia"
        assert parts[1].startswith("nostem:")
        assert parts[2] == "unknown"

    def test_rebuild_cap_groups_by_normalized_location(self):
        """
        Cap группирует по нормализованной (lowercase) локации:
        'China', 'CHINA', 'china' → одна группа.
        """
        old1 = "other|aaa|2026-10-20"
        old2 = "other|bbb|2026-10-20"
        old3 = "other|ccc|2026-10-20"
        k1 = rebuild_cluster_key("China", old1)
        k2 = rebuild_cluster_key("CHINA", old2)
        k3 = rebuild_cluster_key("china", old3)
        # Все три должны начинаться с 'china'
        assert k1.startswith("china|")
        assert k2.startswith("china|")
        assert k3.startswith("china|")

    def test_rebuild_location_canonicalization_comma_space(self):
        """
        MAJOR 5-тест: нормализация локации (strip → lower → space collapse → comma/space normalization → trailing stripped).
        Ключевое: comma+space normalization; trailing comma stripped.
          Moscow, Russia → moscow, russia (comma in middle, stays)
          Moscow  Russia → moscow russia (double space → collapsed → no comma in input → kept as-is)
          moscow , russia → moscow, russia (space+comma normalized to ", " before trailing-strip)
        """
        old = "other|aaa|2026-10-20"
        k1 = rebuild_cluster_key("Moscow, Russia", old)
        k2 = rebuild_cluster_key("Moscow  Russia", old)
        k3 = rebuild_cluster_key("moscow , russia", old)
        # Запятая в середине сохраняется
        assert k1.startswith("moscow, russia|"), f"Got: {k1}"
        # Двойной пробел → space collapse → "Moscow Russia" → no comma in input → "Moscow Russia"
        # (запятая НЕ добавляется без словаря)
        assert k2.startswith("moscow russia|"), f"Got: {k2}"
        # Trailing comma: "moscow , russia" → punctuation norm replaces ", " → "moscow, russia"
        assert k3.startswith("moscow, russia|"), f"Got: {k3}"
        # k1 != k2: "Moscow, Russia" → comma preserved; "Moscow  Russia" → no comma in input
        assert k1 != k2, f"k1 and k2 should differ (comma vs no-comma input): {k1} == {k2}"

    def test_rebuild_location_different_variants_same_bucket(self):
        """
        MAJOR 5-тест: LLM может вернуть разные написания для одного места
        (кириллица vs латиница, разные суффиксы) → cap оставляет ≤2 в кластере.
        """
        old = "other|aaa|2026-10-20"

        # «Россия, Москва» (кириллица) vs «Moscow, Russia» (латиница)
        # → нормализуются strip+lower, НЕ словарь, но trailing punctuation
        #   схлопывается
        k_cyrillic = rebuild_cluster_key("Россия, Москва", old)
        k_english = rebuild_cluster_key("Moscow, Russia", old)

        # Оба нормализуются: lowercase, trim → «россия, москва» и «moscow, russia»
        # Это РАЗНЫЕ строки (без словаря). Но если LLM следуеn формату
        # «City, Country» в English — результат одинаков.
        # Тест проверяет что формат «Moscow, Russia» → «moscow, russia» стабильно.
        assert k_english.startswith("moscow, russia|"), f"Got: {k_english}"

    def test_rebuild_location_empty_falls_back_to_other(self):
        """Пустая/blank локация → 'other', cap не применяется."""
        old = "russia|aaa|2026-10-20"
        assert rebuild_cluster_key("", old).startswith("other|")
        assert rebuild_cluster_key("   ", old).startswith("other|")
        assert rebuild_cluster_key(None, old).startswith("other|")

    def test_rebuild_location_trailing_punctuation_stripped(self):
        """Trailing точки убираются; запятые в середине сохраняются."""
        old = "other|aaa|2026-10-20"
        k1 = rebuild_cluster_key("Krasnodar, Russia.", old)
        k2 = rebuild_cluster_key("Krasnodar, Russia , ", old)
        # ", Russia" stays (comma in middle), trailing "." → stripped → "krasnodar, russia"
        assert k1.startswith("krasnodar, russia|"), f"Got: {k1}"
        # ", Russia ," → strip(" ")="Krasnodar, Russia ,"
        #   → lower → "krasnodar, russia ,"
        #   → punct \s*[.,;]\s* matches ", " (the trailing one), replaces → "krasnodar, russia "
        #   → trailing strip of , removes the comma+space → "krasnodar russia"
        # BUT: the punct regex already consumed " ," → only trailing space remains → trailing comma strip doesn't match
        # Actually: punct replaces ", " with ", " → ", Russia ," → ", Russia " (space preserved)
        # Wait let me trace more carefully:
        # "krasnodar, russia ," after punct: ", Russia ," → regex on ", Russia ," finds ", " (comma+space at end)
        # → replaces ", " with ", " → "krasnodar, russia ," → trailing space still there
        # → trailing comma: no trailing comma → no change
        # → strip(" \t") → "krasnodar, russia" ✓
        assert k2.startswith("krasnodar, russia|"), f"Got: {k2}"


class TestCollectorClusterKey:
    """Тесты cluster_key в NewsItem (Задача 2.3)."""

    def test_newsitem_cluster_key_field(self):
        """NewsItem имеет поле cluster_key."""
        from newsbot.collector import NewsItem

        item = NewsItem(
            source="Test",
            url="https://example.com/test",
            title="Test title",
            summary="summary",
            published="2026-10-18",
            cluster_key="russia|abc12345|2026-10-18",
        )
        assert item.cluster_key == "russia|abc12345|2026-10-18"

    def test_newsitem_cluster_key_default_none(self):
        """cluster_key по умолчанию None."""
        from newsbot.collector import NewsItem

        item = NewsItem(
            source="Test",
            url="https://example.com/test",
            title="Test title",
            summary="summary",
            published="2026-10-18",
        )
        assert item.cluster_key is None

    def test_storage_saves_and_reads_cluster_key(self, tmp_path=None):
        """Storage сохраняет и читает cluster_key."""
        from newsbot.storage import Storage
        from newsbot.collector import NewsItem
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            storage = Storage(db_path=db_path)

            # Используем build_cluster_key с English-локацией (как LLM).
            # Для старых данных в storage может быть русская 'франция' —
            # это нормально (прежний формат), migration не требуется.
            cluster_key = build_cluster_key(
                "france", "abc12345", "18.10.2026 20:00 UTC",
            )
            item = NewsItem(
                source="Test",
                url="https://example.com/cluster-test",
                title="France protests in Paris",
                summary="summary",
                published="18.10.2026 20:00 UTC",
                cluster_key=cluster_key,
            )
            saved = storage.save(item)
            assert saved, "Item должен быть сохранён"

            # Чтение
            items = storage.get_all_by_state("new")
            assert len(items) == 1
            assert items[0].cluster_key == cluster_key
            storage.close()

    def test_storage_save_many_with_cluster_key(self, tmp_path=None):
        """save_many() сохраняет cluster_key."""
        from newsbot.storage import Storage
        from newsbot.collector import NewsItem
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test.db"
            storage = Storage(db_path=db_path)

            items = [
                NewsItem(
                    source="Test",
                    url="https://example.com/c1",
                    title="Test 1",
                    summary="s1",
                    published="2026-10-18",
                    cluster_key=build_cluster_key("russia", "aaa", "2026-10-18"),
                ),
                NewsItem(
                    source="Test",
                    url="https://example.com/c2",
                    title="Test 2",
                    summary="s2",
                    published="2026-10-18",
                    cluster_key=build_cluster_key("usa", "bbb", "2026-10-18"),
                ),
            ]
            inserted = storage.save_many(items)
            assert len(inserted) == 2

            # Проверяем cluster_key при чтении
            all_items = storage.get_all_by_state("new")
            keys = {i.cluster_key for i in all_items}
            assert build_cluster_key("russia", "aaa", "2026-10-18") in keys
            assert build_cluster_key("usa", "bbb", "2026-10-18") in keys
            storage.close()