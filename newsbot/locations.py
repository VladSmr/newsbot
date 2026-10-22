"""Справочник локаций для кластеризации новостей."""
from typing import Optional

LOCATIONS = {
    # Страны
    'russia': 'россия', 'rf': 'россия', 'рф': 'россия', 'russian federation': 'россия', 'russian': 'россия',
    'france': 'франция', 'french': 'франция',
    'ukraine': 'украина', 'украина': 'украина', 'kyiv': 'украина',
    'usa': 'сша', 'united states': 'сша', 'america': 'сша', 'united states of america': 'сша', 'us': 'сша',
    'germany': 'германия', 'german': 'германия',
    'uk': 'британия', 'britain': 'британия', 'england': 'британия', 'british': 'британия',
    'china': 'китай', 'chinese': 'китай',
    'india': 'индия', 'indian': 'индия',
    'japan': 'япония', 'japanese': 'япония',
    'brazil': 'бразилия',
    'israel': 'израиль', 'israeli': 'израиль',
    'iran': 'иран', 'iranian': 'иран',
    'turkey': 'турция', 'turkish': 'турция',
    'poland': 'польша',
    'italy': 'италия',
    'spain': 'испания',
    'canada': 'канада',
    'australia': 'австралия',
    'south africa': 'юар',
    'saudi arabia': 'саудиты', 'saudi': 'саудиты',
    'yemen': 'йемен',
    'haiti': 'гаити', 'haitian': 'гаити',
    # Города
    # NB: 'kiev' определён ТОЛЬКО здесь (город). В блоке стран ранее был
    # дубликат 'kiev': 'украина', который молча перекрывался этой записью
    # (мёртвая запись) — удалён, чтобы не путать при правках. 'kyiv' остаётся
    # в блоке стран (см. выше).
    'paris': 'париж', 'marseille': 'марсель', 'lyon': 'лион',
    # 'москва'/'киев' — именительный падеж кириллицей (в заголовках встречается
    # и он: «Москва», «Киев»); ранее были только косвенные формы.
    'moscow': 'москва', 'москва': 'москва', 'москве': 'москва',
    'москвы': 'москва', 'московская': 'москва',
    'kiev': 'киев', 'киев': 'киев', 'киеве': 'киев', 'киева': 'киев',
    'washington': 'вашингтон',
    'london': 'лондон',
    'berlin': 'берлин',
    'tokio': 'токио', 'tokyo': 'токио',
    'beijing': 'пекин',
    # Регионы РФ (для тестов)
    'белгородской': 'россия', 'белгородская': 'россия', 'белгородской области': 'россия',
    'белгородская область': 'россия',
}


# NIT: набор символов, снимаемых с краёв токена при извлечении локации.
# Включены типографские знаки («ёлочки», тире, многоточие, «умные» кавычки):
# без них «Москва», Москва… и Москва — не матчились.
_TOKEN_STRIP_CHARS = '.,;:!?()[]{}"\'-«»—–…“”‘’'


def extract_location(title: str) -> str:
    """Извлекает локацию из заголовка: n-граммы 3→2→1, страны > города."""
    words = title.split()
    if not words:
        return 'other'

    candidates = []

    # Сканируем n-граммы от длинной к короткой (3 → 2 → 1).
    # Для каждой n-граммы:
    #   - токены не должны склеиваться через пунктуацию
    #   - для 'us' (US) — только если заглавная буква в оригинале и стоит
    #     как отдельное слово без пунктуации (не "helps us")
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            raw_tokens = words[i:i + n]
            clean_tokens = []
            valid = True
            for tok in raw_tokens:
                clean = tok.strip(_TOKEN_STRIP_CHARS)
                if not clean:
                    valid = False
                    break
                if '.' in clean or ',' in clean:
                    valid = False
                    break
                clean_tokens.append(clean.lower())

            if not valid:
                continue

            phrase = ' '.join(clean_tokens)
            if phrase in LOCATIONS:
                # 'us' — только если заглавная US в оригинале и это 1-грамма
                if phrase == 'us':
                    orig = words[i].strip(_TOKEN_STRIP_CHARS)
                    if not orig or orig[0] != orig[0].upper() or orig[0].islower():
                        continue
                candidates.append(LOCATIONS[phrase])

    if not candidates:
        return 'other'

    # Приоритет: страна (список стран) > город > other
    countries = {'россия', 'франция', 'украина', 'сша', 'германия', 'британия',
                 'китай', 'индия', 'япония', 'бразилия', 'израиль', 'иран',
                 'турция', 'польша', 'италия', 'испания', 'канада', 'австралия',
                 'юар', 'саудиты', 'йемен', 'гаити'}
    for c in candidates:
        if c in countries:
            return c
    return candidates[0]


def build_cluster_key(
    location: str, stem: Optional[str], published: Optional[str],
    title: Optional[str] = None,
) -> str:
    """
    Minor 6-фикс: единая реализация сборки cluster_key.
    Публичный helper — вызывается из collector и из locations.cluster_key().

    Per-title fallback (Variant B, CRITICAL-1): при пустом stem_hash
    (emoji-only, пустой normalize, не-Latin/Cyrillic заголовки) хешируется
    СЫРОЙ title (без normalize) — уникальный ключ на exact-заголовок.
    Осознанный выбор: number-stripping нормализации склеивал бы "AI 2024"
    и "AI 2025" в один cluster, хотя это разные новости. Сырой title
    гарантирует разные хеши для разных заголовков.
    Это гарантирует, что разные короткие/эмодзи заголовки в одной локации
    НЕ сливаются в один cluster_key и не блокируются domain-дедупом.

    Для items без title (заголовок не определён или пуст) возвращается
    nostem:<md5(empty bytes)>|day. Это НЕ отдельная «некэпируемая» категория:
    ключ, как и любой другой, кэпится _cap_by_location() по parts[0]
    (локации) и дню публикации. Из локационного cap исключаются только
    items, у которых location == 'other' (см. pipeline._cap_by_location);
    domain-дедуп по url применяется к nostem-ключам наравне с остальными.
    stem_hash уже возвращает None при пустом/не-Latin/Cyrillic normalize
    (см. stemmer.py), и в этом случае мы переходим в ветку nostem
    вместо возврата None.
    """
    import hashlib
    pub_day = published[:10] if (published and len(published) >= 10) else 'unknown'
    if stem:
        return f"{location}|{stem}|{pub_day}"
    else:
        # Emoji-only или empty normalize → per-title hash по исходному title.
        # Без number-stripping: разные заголовки "AI 2024" и "AI 2025" дают
        # РАЗНЫЕ hash'и, что предотвращает склейку разных новостей.
        # Item без title → md5(b'') (nostem:hash из пустых bytes).
        if title:
            title_hash = hashlib.md5(title.encode()).hexdigest()[:12]
        else:
            title_hash = hashlib.md5(b'').hexdigest()[:12]
        return f"{location}|nostem:{title_hash}|{pub_day}"


def cluster_key(item) -> str:
    """
    Minor 6-фикс: wrapper над build_cluster_key для обратной совместимости.
    Все вызовы в production идут через collector.build_cluster_key().
    """
    from .stemmer import stem_hash
    location = extract_location(item.title)
    stem = stem_hash(item.title)
    return build_cluster_key(location, stem, item.published, title=item.title)


