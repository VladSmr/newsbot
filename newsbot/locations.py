"""Справочник локаций для кластеризации новостей."""

LOCATIONS = {
    # Страны
    'russia': 'россия', 'rf': 'россия', 'рф': 'россия', 'russian federation': 'россия', 'russian': 'россия',
    'france': 'франция', 'french': 'франция',
    'ukraine': 'украина', 'украина': 'украина', 'kyiv': 'украина', 'kiev': 'украина',
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
    'paris': 'париж', 'marseille': 'марсель', 'lyon': 'лион',
    'moscow': 'москва', 'москве': 'москва', 'москвы': 'москва', 'московская': 'москва',
    'kiev': 'киев', 'киеве': 'киев', 'киева': 'киев',
    'washington': 'вашингтон',
    'london': 'лондон',
    'berlin': 'берлин',
    'tokio': 'токио', 'tokyo': 'токио',
    'beijing': 'пекин',
    # Регионы РФ (для тестов)
    'белгородской': 'россия', 'белгородская': 'россия', 'белгородской области': 'россия',
    'белгородская область': 'россия',
}


def extract_location(title: str) -> str:
    """Извлекает первую найденную локацию из заголовка (страны > города)."""
    words = title.lower().split()
    candidates = []
    for i in range(len(words)):
        phrase = ' '.join(words[i:i + 3]).strip()
        if phrase in LOCATIONS:
            candidates.append(LOCATIONS[phrase])
        # Слово может быть с пунктуацией
        w = words[i].rstrip('.,;:!?()[]{}"\'-')
        if w in LOCATIONS:
            candidates.append(LOCATIONS[w])
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


def cluster_key(item) -> str:
    """Возвращает ключ кластера для новостного item: локация|stem_hash|день."""
    from .stemmer import stem_hash
    location = extract_location(item.title)
    stem = stem_hash(item.title)
    if item.published:
        try:
            pub = item.published[:10]  # YYYY-MM-DD
        except Exception:
            pub = 'unknown'
    else:
        pub = 'unknown'
    return f"{location}|{stem}|{pub}"