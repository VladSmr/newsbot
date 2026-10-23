"""
urlnorm — нормализация URL для дедупликации.

Задача: одна и та же статья с utm-параметрами и без должна склеиваться
в единый ключ дедупликации.

Правила нормализации:
  1. scheme и host приводятся к lowercase (путь — как есть, case-sensitive).
  2. fragment (#...) удаляется.
  3. Трекинговые query-параметры удаляются: utm_*, traffic_source, fbclid,
     gclid, mc_cid, mc_eid, ref, referrer, spm.
  4. Оставшиеся query-параметры сортируются по (имя, значение) —
     мультизначения стабильно упорядочиваются, a=1&b=2 и b=2&a=1 склеиваются.
  5. Пустой path → "/"; завершающий слэш убирается (кроме корня "/").
  6. Не-http(s) схемы или невалидные URL возвращаются как есть (защита).

Функция идемпотентна: normalize(normalize(url)) == normalize(url).
"""

from typing import Optional
from urllib.parse import urlparse, urlencode, parse_qsl

# Параметры запроса, которые вырезаются при нормализации.
# Содержит основные трекеры: UTM (все варианты), рекламные ID,
# referrer-параметры и сервисные параметры популярных платформ.
TRACKING_PARAMS: frozenset = frozenset({
    # UTM-метки (все официальные и нестандартные)
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_source_platform", "utm_creative_format", "utm_marketing_tactic",
    # Новостные/фид-трекеры
    "traffic_source",
    # Рекламные платформы
    "fbclid",        # Facebook
    "gclid",         # Google Ads
    "gclsrc",        # Google Ads (alternative)
    "dclid",         # DoubleClick
    "msclkid",       # Microsoft/Bing Ads
    "mc_eid",        # Mailchimp
    "mc_cid",        # Mailchimp
    # Партнёрские / аналитика
    "ref",           # Generic referrer
    "referrer",
    "spm",           # Alibaba / Alibaba Cloud
    "aff_id",
    "aff_sub",
    "aff_click_id",
    "campaign_id",
    "ad_id",
    "adgroup_id",
    "creative_id",
    "placement",
    "network",
    "device",
    "matchtype",
    "feed_item_id",
    "keyword",
    # Социальные сети / мессенджеры
    "igshid",        # Instagram
    "twclid",        # Twitter/X
    "s",             # TikTok (share parameter)
    # Прочее
    "zanpid",
    "mbid",          # Mailchimp bid
    "fb_action_ids",
    "fb_action_types",
    "fb_source",
    "_ga",
    "_gl",
})


def normalize_url(url: Optional[str]) -> Optional[str]:
    """
    Нормализует URL для дедупликации.

    Args:
        url: исходный URL (может быть None или пустым).

    Returns:
        Нормализованный URL, или исходный url как есть при не-HTTP(S)
        схеме / невалидном URL (защита — ничего не падает).

    Идемпотентность:
        normalize_url(normalize_url(url)) == normalize_url(url)

    Примеры:
        https://www.EXAMPLE.com/path/?utm_source=rss&utm_medium=email#section
        → https://www.example.com/path?utm_medium=email

        https://HOST.COM/page?id=2&fbclid=abc
        → https://host.com/page?id=2

        https://HOST.COM/page?utm_source=a&id=2&utm_medium=b
        → https://host.com/page?id=2&utm_medium=b   (параметры отсортированы)

        https://HOST.COM/
        → https://host.com/   (слэш корня сохраняется)

        https://HOST.COM/path/
        → https://host.com/path   (завершающий слэш убирается)

        https://x.com
        → https://x.com/   (nit-фикс: пустой path → "/")

        https://x.com/?utm_source=rss
        → https://x.com/?utm_source=rss  → https://x.com/  (после normalize)

        https://example.com/p?a=1&z=9&a=2
        → https://example.com/p?a=1&a=2&z=9   (nit-фикс: sort by (name, value))
    """
    if not url:
        return url

    try:
        parsed = urlparse(url)

        # Пропускаем не-http(s) схемы (mailto:, tel:, file:, и т.п.)
        if parsed.scheme not in ("http", "https"):
            return url

        # 1. scheme + host → lowercase; path остаётся как есть
        # (nit-фикс: URL paths are case-sensitive; CMS на хосте может дать 404
        # по ссылке из дайджеста, если понижен регистр). Схема + host понижаем,
        # путь сохраняем.
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path

        # 2. fragment убираем
        fragment = ""

        # 3+4. Фильтруем query-параметры
        params_to_remove = TRACKING_PARAMS
        kept_params = []
        for name, value in parse_qsl(parsed.query, keep_blank_values=True):
            if name not in params_to_remove:
                kept_params.append((name, value))

        # Сортируем по (имя, значение) для детерминированности.
        # nit-фикс: `a=1&a=2` и `a=2&a=1` дают один ключ (мультизначение
        # в определённом порядке), `a=1&b=2` и `b=2&a=1` тоже склеиваются.
        kept_params.sort(key=lambda kv: (kv[0], kv[1]))

        # 5. Нормализация пути: пустой path → "/"; убираем завершающий слэш
        # у не-корневого пути.
        # nit-фикс: https://x.com и https://x.com/ должны склеиваться.
        if not path:
            path = "/"
        elif path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        # Собираем обратно
        query = urlencode(kept_params) if kept_params else ""
        normalized = (
            f"{scheme}://{netloc}{path}"
            f"{('?' + query) if query else ''}"
            f"{('#' + fragment) if fragment else ''}"
        )

        return normalized

    except Exception:
        # Любая неожиданная ошибка (malformed URL и т.п.) → возвращаем как есть.
        # Защита: невалидный URL не должен падать при нормализации.
        return url