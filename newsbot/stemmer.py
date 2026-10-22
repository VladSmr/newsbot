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

# Плюралисы на -us (лат./греч. основы): снимаем -es, а не только -s.
# Регрессия волны 2: общее правило «-es = немое e + s, снимаем только -s»
# верно для case/response/house (основы на -se/-use), но для основ на -us
# давало viruse/statuse/buse — расходилось с единственным числом
# (virus/status/bus) и ломало склейку перепечаток. Набор явный и узкий:
# основы на -use/-ous (house→house, purpose→purpose) остаются нетронутыми,
# т.к. там -e действительно часть корня.
_ES_TO_US_PLURALS = frozenset({
    'viruses',      # virus
    'statuses',     # status
    'buses',        # bus
    'censuses',     # census
    'consensuses',  # consensus
    'crocuses',     # crocus
})


def _stem_word(w: str) -> str:
    """
    Suffix stripper с защитой от over-stemming.
    
    Правила (порядок важен):
      -tion     → бесплатно (4 буквы)
      -ing      → если >=5 букв и перед ing согласная (иначе "king", "ring" etc.)
      -ment     → бесплатно
      -ness     → бесплатно
      -able/-ible/-ful/-ous/-ive → если >=6 букв
      -ed       → если >=4 и перед ed согласная; откатываем удвоение (stopped→stop),
                  восстанавливаем немое -e после гласной+s (raised→raise, used→use)
      -ies      → если >=5 и перед ies согласная → ies→y (rallies→rally)
      -es       → снимаем -es ТОЛЬКО у основ на -ss/-ch/-sh/-x/-z (passes→pass,
                  watches→watch, boxes→box); у плюралисов на -us (viruses→virus,
                  buses→bus) тоже снимаем -es; у остальных — обычное -s
                  (cases→case, raises→raise, notes→note, rates→rate)
      -s        → если >=3 и перед s согласная (без e: protests, clashes→clash)
    
    Примеры:
      protests  → protest (s убран, protest ≥5 букв)
      rallies   → rally   (ies→y, rally ≥4 букв)
      clashes   → clash   (s убран, clash ≥4 букв)
      stopped   → stop    (ed убран, удвоение откачено)
      paris     → paris   (s НЕ убран: гласная 'i' перед s)
      rates     → rate    (es убран как обычное -s, не rate→rat)
      raised    → raise   (ed убран + восстановлено немое e: rais→raise)
      cases     → case    (немое -e сохраняется: снят только -s, а не -es)
      responses → response (то же правило)
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
        # восстанавливаем немое -e: raised→raise (rais), used→use (us).
        # Без этого raised→rais, а raise→raise — расходятся по стемам.
        # Исключение: основы на -us (focus, status) — там -ed снимается как есть.
        elif (
            len(base) >= 2
            and base.endswith('s')
            and base[-2] in VOWELS
            and not (len(base) > 2 and base.endswith('us'))
        ):
            base += 'e'
        return base
    # -ies: rallies → rally, policies → policy
    if w.endswith('ies') and len(w) > 4 and w[-4] in CONSONANTS:
        return w[:-3] + 'y'
    # -es: снимаем -es ТОЛЬКО у основ на шипящие/свистящие (-ss/-ch/-sh/-x/-z):
    # passes→pass, watches→watch, boxes→box, clashes→clash, classes→class.
    # У остальных -es — это немое -e + множественное -s: снимаем только -s
    # (cases→case, raises→raise, responses→response, notes→note, rates→rate).
    # Раньше -es снимался с любой согласной: cases→cas, notes→not, rates→rat —
    # и case/cases, rate/rates расходились по кластерам, из-за чего
    # domain-дедуп не склеивал перепечатку.
    if w.endswith('es') and len(w) > 3:
        # -us-плюралисы: снимаем -es целиком (viruses→virus, buses→bus)
        if w in _ES_TO_US_PLURALS:
            return w[:-2]
        if w.endswith(('sses', 'ches', 'shes', 'xes', 'zzes')):
            return w[:-2]
        return w[:-1]
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
    # Стоп-слова фильтруем ДО стемминга: иначе does→doe / has→ha / was→wa
    # перестают совпадать со STOP_WORDS и попадают в stem_hash мусором.
    words = [w for w in words if w not in STOP_WORDS]
    words = [_stem_word(w) for w in words]
    # Повторная проверка оставлена как страховка: стем может совпасть
    # со стоп-словом (напр. 'doing'→'doing' не совпадёт, но правило дешёвое).
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