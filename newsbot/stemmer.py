"""Простой stemmer для кластеризации новостей. CPU, без зависимостей."""
import re
from typing import Optional

STOP_WORDS = {
    # EN
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'that', 'which', 'who', 'what', 'where', 'when', 'how', 'why', 'this', 'these',
    'those', 'its', 'not', 'no', 'so', 'if', 'then', 'than', 'also', 'can', 'just',
    # RU
    'в', 'на', 'по', 'из', 'с', 'и', 'к', 'за', 'о', 'у', 'от', 'до', 'при', 'для',
    'не', 'что', 'как', 'это', 'то', 'оно', 'она', 'они', 'мы', 'вы', 'ты', 'он',
    'был', 'была', 'было', 'были', 'есть', 'быть', 'иметь', 'имеет', 'имели',
    'который', 'которая', 'которое', 'которые', 'этот', 'эта', 'эти',
}

CONSONANTS = frozenset('bcdfghjklmnpqrstvwxyz')
VOWELS = frozenset('aeiou')


def _stem_word(w: str) -> str:
    """
    Suffix stripper с защитой от over-stemming.
    
    Правила (порядок важен):
      -tion     → бесплатно (4 буквы)
      -ing      → если >=5 букв и перед ing согласная (иначе "king", "ring" etc.)
      -ment     → бесплатно
      -ness     → бесплатно
      -able/-ible/-ful/-ous/-ive → если >=6 букв
      -ed       → если >=4 и перед ed согласная; откатываем удвоение (stopped→stop)
      -ies      → если >=5 и перед ies согласная → ies→y (rallies→rally)
      -es       → если >=4 и перед es согласная (stopped, cases→case)
      -s        → если >=3 и перед s согласная (без e: protests, clashes→clash)
    
    Примеры:
      protests  → protest (s убран, protest ≥5 букв)
      rallies   → rally   (ies→y, rally ≥4 букв)
      clashes   → clash   (s убран, clash ≥4 букв)
      stopped   → stop    (ed убран, удвоение откачено)
      paris     → paris   (s НЕ убран: гласная 'i' перед s)
    """
    if len(w) <= 2:
        return w

    # -tion
    if w.endswith('tion') and len(w) > 5:
        return w[:-4]
    # -ing: "king", "ring", "thing" не трогаем
    if w.endswith('ing') and len(w) > 4 and w[-4] in CONSONANTS:
        return w[:-3]
    # -ment
    if w.endswith('ment') and len(w) > 5:
        return w[:-4]
    # -ness
    if w.endswith('ness') and len(w) > 5:
        return w[:-4]
    # -able / -ible / -ful / -ous / -ive
    for suff in ('able', 'ible', 'ful', 'ous', 'ive'):
        if w.endswith(suff) and len(w) > len(suff) + 2:
            return w[:-len(suff)]
    # -ied: rallies → rally (not ralle)
    if w.endswith('ied') and len(w) > 4 and w[-4] in CONSONANTS:
        return w[:-3] + 'y'
    # -ed: stopped → stop (откатываем удвоение), attacked → attack
    if w.endswith('ed') and len(w) > 3 and w[-3] in CONSONANTS:
        base = w[:-2]  # remove 'ed'
        # откатываем удвоение: stopped→stop, attacked→attack
        if len(base) > 2 and base[-1] == base[-2] and base[-1] not in VOWELS:
            base = base[:-1]
        return base
    # -es: cases → case, passes → pass
    if w.endswith('es') and len(w) > 3 and w[-3] in CONSONANTS:
        return w[:-2]
    # -s: protests, clashes → protest, clash (без e)
    if w.endswith('s') and len(w) > 2 and w[-2] in CONSONANTS:
        return w[:-1]

    return w


def normalize(text: str) -> str:
    """Нормализует заголовок: числа, стоп-слова, суффиксы, сортировка."""
    text = re.sub(r'\d+([.,]\d+)?', ' ', text.lower())
    # Minor 7-фикс: добавлены укр. буквы і/ї/є/ґ (U+0456/U+0457/U+0454/U+0491)
    text = re.sub(r'[^a-zA-Z\u0430-\u044f\u0451\u0456\u0457\u0454\u0491\s]', ' ', text)
    words = text.split()
    words = [_stem_word(w) for w in words]
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(sorted(words))


def stem_hash(text: str) -> Optional[str]:
    """
    Хеш нормализованного stem для кластеризации.
    Minor 7-фикс: при пустом normalize (empty stem → md5("") коллизия)
    возвращает None. Вызывающий код должен обработать sentinel.
    """
    import hashlib
    norm = normalize(text)
    if not norm:
        return None
    return hashlib.md5(norm.encode()).hexdigest()[:8]