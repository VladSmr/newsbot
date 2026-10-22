"""
Дайджест отложенных новостей (state='deferred').

Формирует текстовые сообщения для отправки владельцу:
  — собирает deferred items (top-N по llm_score DESC)
  — разбивает на блоки по max_items_per_message и лимиту 4096 символов
  — HTML-экранирование всех пользовательских данных
  — URL-whitelist: только http:// / https://

At-least-once: set_digested вызывается ТОЛЬКО для items,
чьи сообщения реально отправлены (частичный сбой → отправленные → digested,
остаток → deferred, retry в следующем окне).
"""
import html
import logging
import re
from datetime import datetime, timezone
from typing import List, Tuple

from .delivery import Delivery, format_time_utc
from .delivery.telegram import is_valid_http_url

# Тип для кнопок одной новости в дайджесте
DigestItemButtons = List[dict]  # [{url, target_id}, ...]
# Тип для per-message item_buttons
DigestMessageButtons = List[List[dict]]  # List of item_buttons per message
from .storage import NewsItem, Storage

logger = logging.getLogger("newsbot.digest")

# Лимит Telegram-сообщения
_TG_MESSAGE_MAX = 4096
# Резерв на футер + запас
_MARGIN = 20
# Максимальная длина URL в footer (ПОСЛЕ escape).
# M1'-фикс: cap по ESCAPED (не RAW), т.к. "&"/'"' раздуваются в 6 раз.
# Максимальный escaped overhead footer: "<a href=\"\">🔗</a>" ≈ 17 + url_esc + time_esc.
# Максимальный footer с time≈20: 17+500+20 = 537 → footer << 4096.
_MAX_FOOTER_URL_ESC_LEN = 500


# =============================================================================
# render (до HTML-сборки, работаем с raw-строками)
# =============================================================================

def _truncate_raw_summary(raw: str, max_chars: int) -> str:
    """
    Обрезает raw-текст до max_chars по границе слова.
    Вызывается ДО html.escape().
    """
    if len(raw) <= max_chars:
        return raw
    truncated = raw[:max_chars]
    m = re.search(r'\s+\S*$', truncated)
    if m:
        truncated = truncated[:m.start()].rstrip()
    else:
        truncated = truncated.rstrip()
    return truncated + "…"


def _render_item_raw(item: NewsItem) -> Tuple[str, str, str, str]:
    """
    Возвращает четвёрку raw-строк (ДО HTML-escape) для одного item:
      (title_raw, source_raw, body_raw, footer_raw)

    footer_raw = "{time_str} 🔗" если URL валидный, иначе "{time_str}"
    """
    title_raw = item.title or "—"
    # заголовок ограничен до 250 символов
    title_raw = _truncate_raw_summary(title_raw, 250)

    source_raw = item.source or "—"
    # M1-фикс: source ограничивается до 100 символов RAW
    # (чтобы escape не раздул overhead и не сломал бюджет)
    source_raw = _truncate_raw_summary(source_raw, 100)

    # body: приоритет llm_summary → summary, до 500 символов
    raw_body = (item.llm_summary or item.summary or "").strip()
    raw_body = _truncate_raw_summary(raw_body, 500)

    # время
    pub_parsed = getattr(item, "published_parsed", None)
    time_str = format_time_utc(pub_parsed, item.fetched_at)

    # URL-whitelist: использует общий helper (согласован с telegram.py).
    # is_valid_http_url: схема http/https, нет control-символов, нет whitespace,
    # нет leading/trailing, длина ≤ 1024.
    url_clean = is_valid_http_url(item.url or "")

    if url_clean:
        url_esc = html.escape(url_clean)
        if len(url_esc) > _MAX_FOOTER_URL_ESC_LEN:
            # Minor-фикс: НЕ обрезаем URL-prefix (дает битую ссылку).
            # Вместо этого отдаём футер БЕЗ <a> — валидный HTML, без ссылки.
            # Пользователь увидит время и текст, но не получит битой ссылки.
            logger.debug(
                "URL слишком длинный (%d escaped, лимит %d) — "
                "футер без ссылки для %s",
                len(url_esc), _MAX_FOOTER_URL_ESC_LEN, url_clean[:60],
            )
            footer = time_str
        else:
            footer = f'{time_str} <a href="{url_esc}">🔗</a>'
    else:
        footer = time_str
    return title_raw, source_raw, raw_body, footer


# =============================================================================
# build_digest_messages
# =============================================================================

def build_digest_messages(
    items: List[NewsItem],
    max_items_per_message: int = 20,
    local_now: datetime = None,
) -> Tuple[List[Tuple[str, List[NewsItem]]], List[NewsItem], List[str]]:
    """
    Формирует список сообщений дайджеста.

    Каждое сообщение содержит ≤ max_items_per_message новостей
    и укладывается в _TG_MESSAGE_MAX символов.

    Стратегия разбиения:
      1. count limit: > max_items_per_message → новое сообщение
      2. budget limit: entry не влезает → flush + start fresh
      3. oversized entry: single entry слишком большой → summary
         урезается ДО escape так, чтобы гарантированно влезть

    Формат сообщения:
      📰 Дайджест — N из TOTAL новостей — <дата/время>
      ─────────────────────────────
      <b>Title — Source</b>
      body
      footer

    MAJOR 2-фикс: per-item try/except при рендеринге — битый item
    выпадает из groups, его url добавляется в build_failed_urls.

    Args:
        items: отсортированный список (score DESC, fetched_at ASC).
        max_items_per_message: новостей на сообщение (минимум 1).
        local_now: текущее время для заголовка.

    Returns:
        Tuple[List[Tuple[str, List[NewsItem]]], List[NewsItem], List[str]]:
          message_groups: список (html_message, items_in_message).
          all_included_items: плоский список всех включённых items
                              (для отладки / отчёта).
          build_failed_urls: urls items, выпавших из-за исключения при
                              рендеринге. Эти items помечаются 'failed'.
        Пустой items → ([], [], []).
    """
    if local_now is None:
        local_now = datetime.now(timezone.utc).astimezone()

    max_items_per_message = max(1, max_items_per_message)

    time_str = local_now.strftime("%d.%m.%Y %H:%M")

    if not items:
        return [], [], []

    # Каждый элемент: (title_raw, source_raw, body_raw, footer_raw, item).
    # MAJOR 2-фикс: per-item try/except — битый item не падает на весь дайджест.
    build_failed_urls: List[str] = []
    rendered: List[Tuple[str, str, str, str, NewsItem]] = []
    for it in items:
        try:
            rendered.append((*_render_item_raw(it), it))
        except Exception as e:
            logger.warning(
                "build_digest: item '%s' исключение при рендеринге: %s — "
                "исключается из дайджеста, будет помечен 'failed'",
                (it.title or "")[:60], e,
            )
            build_failed_urls.append(it.url)

    # Nit-фикс: «N из TOTAL» считает только фактически отрендеренные items.
    # Выпавшие при рендеринге (build_failed_urls) в знаменатель не входят —
    # иначе сплошная нумерация даёт «29 из 30» при 29 отрендеренных.
    total = len(rendered)

    def _entry_size(title_raw: str, source_raw: str, body_raw: str,
                    footer_raw: str) -> int:
        """Верхняя оценка размера entry после HTML-escape."""
        return (
            len(f"<b>{html.escape(title_raw)} — {html.escape(source_raw)}</b>")
            + 2  # \n\n
            + len(html.escape(body_raw))
            + 2  # \n\n
            + len(footer_raw)
        )

    message_groups: List[Tuple[str, List[NewsItem]]] = []
    all_included: List[NewsItem] = []

    # Текущее сообщение
    cur_entries: List[Tuple[str, NewsItem]] = []   # (entry_html, item)
    cur_raw: List[Tuple[str, str, str, str]] = []  # (title, source, body, footer)
    msg_start_index: int = 0   # 1-based start item number для текущего msg

    def _make_header(start: int, end: int) -> str:
        if start == end:
            return (
                f"📰 Дайджест — {start} из {total} новостей — {time_str}\n"
                + "─" * 40 + "\n"
            )
        return (
            f"📰 Дайджест — {start}–{end} из {total} новостей — {time_str}\n"
            + "─" * 40 + "\n"
        )

    def _emit() -> None:
        """Формирует и сохраняет текущее сообщение."""
        nonlocal message_groups, all_included, cur_entries, cur_raw, msg_start_index
        if not cur_entries:
            return
        start = msg_start_index + 1
        end = msg_start_index + len(cur_entries)
        header = _make_header(start, end)
        body = "\n\n".join(html for html, _ in cur_entries)
        msg = header + "\n" + body
        message_groups.append((msg, [item for _, item in cur_entries]))
        all_included.extend(item for _, item in cur_entries)
        cur_entries = []
        cur_raw = []
        msg_start_index = end

    def _assemble_entry(title_r: str, source_r: str, body_r: str,
                       footer_r: str, usable: int) -> str:
        """
        Собирает один entry (HTML) в бюджет usable символов.
        body урезается так, чтобы entry ГАРАНТИРОВАННО влёз в usable.

        M1-фикс (N3): использует бинарный поиск по RAW body, чтобы
        фактическая ESCAPED-длина (после html.escape) <= usable.
        Безопасные границы — режет только на границе слова (non-space boundary).
        """
        title_esc = html.escape(title_r)
        source_esc = html.escape(source_r)

        # overhead: <b> + </b> + " — " + footer + \n\n + margin
        # source_esc вычисляется ДО поиска — не меняется при урезании body
        overhead = (
            len(f"<b>{title_esc} — {source_esc}</b>")
            + 2
            + len(footer_r)
            + _MARGIN
        )
        budget = usable - overhead
        if budget < 0:
            budget = 0

        # Бинарный поиск: находим max_body_raw → после escape ≤ budget
        # Безопасная граница — режем по word boundary если возможно
        body_len = len(body_r)
        if body_len == 0:
            return f"<b>{title_esc} — {source_esc}</b>\n\n{footer_r}"

        lo, hi = 0, body_len
        # Итерируем: уменьшаем hi пока escaped > budget
        while lo < hi:
            mid = (lo + hi + 1) // 2   # верхняя середина
            candidate = _truncate_raw_summary(body_r, mid)
            escaped_len = len(html.escape(candidate))
            if escaped_len <= budget:
                lo = mid
            else:
                hi = mid - 1

        body_trunc = _truncate_raw_summary(body_r, lo)
        body_esc = html.escape(body_trunc)

        return f"<b>{title_esc} — {source_esc}</b>\n{body_esc}\n{footer_r}"

    for title_r, source_r, body_r, footer_r, item in rendered:
        if len(cur_entries) >= max_items_per_message:
            _emit()

        if cur_entries:
            # Есть накопленные: считаем бюджет
            header = _make_header(msg_start_index + 1, msg_start_index + len(cur_entries) + 1)
            used = len(header)
            for t, s, b, f in cur_raw:
                used += _entry_size(t, s, b, f) + 2
            # entry + \n\n separator
            entry_est = _entry_size(title_r, source_r, body_r, footer_r)
            if used + 2 + entry_est > _TG_MESSAGE_MAX:
                _emit()

        if not cur_entries:
            # Fresh message: считаем budget для одного entry
            header = _make_header(msg_start_index + 1, msg_start_index + 1)
            usable = _TG_MESSAGE_MAX - len(header) - _MARGIN
            entry_html = _assemble_entry(
                title_r, source_r, body_r, footer_r, usable
            )
            cur_entries.append((entry_html, item))
            cur_raw.append((title_r, source_r, body_r, footer_r))
        else:
            # Добавляем к накопленным
            cur_entries.append(
                (_assemble_entry(title_r, source_r, body_r, footer_r,
                                 usable=9999), item)
            )
            cur_raw.append((title_r, source_r, body_r, footer_r))

    _emit()
    return message_groups, all_included, build_failed_urls


# =============================================================================
# send_digest (internal — используется pipeline.run_digest)
# =============================================================================

def _send_digest_with_targets(
    delivery: Delivery,
    groups: List[Tuple[str, List[NewsItem]]],
    item_buttons_per_msg: List[List[dict]],
) -> Tuple[int, List[str], List[Tuple[str, List[NewsItem]]]]:
    """
    Отправляет группы дайджеста через delivery.send_digest.
    Возвращает (sent_count, sent_urls, sent_groups).

    sent_groups — группы которые реально отправлены
    (нужно pipeline для логирования / отладки).

    При частичном сбое — отправленные → digested, остаток → deferred
    (at-least-once: retry в следующем окне).

    M1-фикс: try/except — ВНУТРИ цикла по сообщениям (как в send_text-ветке).
    Исключение на одном сообщении НЕ обрывает отправку остальных:
    ошибка логируется warning'ом с номером/URL, цикл продолжается.

    N1: send_digest от delivery — возвращает bool (succ/fail).
    N3: лимит 4096 проверен на уровне build_digest_messages;
        консервативно: если текст всё равно >4096 — логируем warning.
    """
    sent_urls: List[str] = []
    sent_groups: List[Tuple[str, List[NewsItem]]] = []

    # Minor-фикс: zip молча обрезает хвост при рассинхроне длин.
    # Логируем рассинхрон и дополняем пустыми кнопками, чтобы
    # ни одна группа не потерялась.
    n_groups = len(groups)
    n_buttons = len(item_buttons_per_msg)
    if n_groups != n_buttons:
        logger.warning(
            "Дайджест: рассинхрон groups=%d vs item_buttons_per_msg=%d — "
            "%s",
            n_groups, n_buttons,
            "дополняем пустыми кнопками для лишних групп"
            if n_groups > n_buttons
            else "обрезаем лишние кнопки",
        )
        if n_groups > n_buttons:
            item_buttons_per_msg = list(item_buttons_per_msg) + [
                [] for _ in range(n_groups - n_buttons)
            ]
        else:
            item_buttons_per_msg = list(item_buttons_per_msg[:n_groups])

    for i, ((msg, group_items), item_buttons) in enumerate(
        zip(groups, item_buttons_per_msg), 1
    ):
        if not msg.strip():
            continue

        # N3: консервативный check на превышение 4096
        if len(msg) > _TG_MESSAGE_MAX:
            logger.warning(
                "Дайджест msg[%d/%d]: %d символов (>%d лимит)",
                i, n_groups, len(msg), _TG_MESSAGE_MAX,
            )

        # M1-фикс: try/except ВНУТРИ цикла — падение одного сообщения
        # не отменяет отправку остальных.
        try:
            ok = delivery.send_digest(msg, item_buttons)
        except Exception as e:
            logger.warning(
                "Дайджест msg[%d/%d]: исключение при отправке (%s): %s — "
                "%d items остаются deferred, остальные сообщения отправляются",
                i, n_groups,
                group_items[0].url if group_items else "—",
                e, len(group_items),
            )
            continue

        if ok:
            sent_urls.extend(item.url for item in group_items)
            sent_groups.append((msg, group_items))
            logger.debug(
                "Дайджест msg[%d/%d]: ok (%d items)",
                i, n_groups, len(group_items),
            )
        else:
            logger.warning(
                "Дайджест msg[%d/%d]: send_digest=False — "
                "%d items остаются deferred",
                i, n_groups,
                sum(len(it) for _, it in groups[i - 1:]),
            )
            # continue: остальные сообщения дайджеста должны уйти
            # (at-least-once: отправленные → digested, этот item → deferred,
            # retry в следующем окне)

    if not sent_urls:
        logger.info("Дайджест: ни одно сообщение не доставлено.")
        return 0, [], []

    return len(sent_urls), sent_urls, sent_groups


# =============================================================================
# run_digest
# =============================================================================

def run_digest(
    storage: Storage,
    delivery: Delivery,
    max_items: int = 100,
    max_items_per_message: int = 20,
    item_buttons_per_msg: List[List[dict]] = None,
    groups: List[Tuple[str, List[NewsItem]]] = None,
    failed_urls_in: List[str] = None,
) -> Tuple[int, List[str], List[str]]:
    """
    Выполняет дайджест: собирает deferred → формирует сообщения →
    отправляет → помечает digested.

    at-least-once: при частичном сбое отправленные items → digested,
    остаток → deferred (retry в следующем окне).

    m2-фикс: если groups передан — использует их напрямую (единый источник
    с pipeline, без пересчёта). Иначе — строит сам (обратная совместимость).

    MAJOR 2-фикс: run_digest возвращает 3-tuple, включая all_failed_urls —
    urls items, исключённых из отправки (per-item исключение при рендеринге
    ИЛИ при создании feedback-target). Pipeline помечает их 'failed'.

    Args:
        storage: экземпляр Storage
        delivery: экземпляр Delivery
        max_items: лимит deferred items для дайджеста
        max_items_per_message: новостей на одно сообщение
        item_buttons_per_msg: кнопки для каждого сообщения
                               List[List[dict]] — кнопки для items в каждом message.
                               Если None — вызывается send_text (обратная совместимость).
        groups: готовые группы сообщений (msg, items).
                Если None — строится из storage.get_deferred_items
                (обратная совместимость для внешних вызовов).
        failed_urls_in: urls items, исключённых ПЕРЕД вызовом run_digest
                        (битый feedback-target). Объединяется с build_failed_urls.

    Returns:
        Tuple[int, List[str], List[str]]: (sent_count, sent_urls, all_failed_urls)
            sent_count: число отправленных items (помечены digested).
            sent_urls: их URLs.
            all_failed_urls: urls всех items, исключённых из отправки
                             (рендеринг + feedback-target + build_failed).
                             Pipeline помечает их 'failed'.
    """
    # m2: groups может быть передан из pipeline (единый источник).
    # При отсутствии — строим сами (old calls, тесты).
    if groups is None:
        items = storage.get_deferred_items(limit=max_items)
        if not items:
            logger.info("Дайджест: нет отложенных новостей.")
            all_failed = list(failed_urls_in) if failed_urls_in else []
            return 0, [], all_failed
        logger.info("Дайджест: собрано %d отложенных новостей (лимит %d)",
                    len(items), max_items)
        groups, all_included, build_failed_urls = build_digest_messages(
            items,
            max_items_per_message=max_items_per_message,
        )
    else:
        all_included = [item for _, grp in groups for item in grp]
        build_failed_urls = []

    # Собираем все failed_urls: build_failed (рендеринг) + failed_urls_in (feedback)
    all_failed_urls: List[str] = list(build_failed_urls)
    if failed_urls_in:
        for url in failed_urls_in:
            if url not in all_failed_urls:
                all_failed_urls.append(url)

    if not groups:
        logger.info("Дайджест: нет сообщений для отправки.")
        return 0, [], all_failed_urls

    logger.info("Дайджест: %d сообщений, %d items", len(groups), len(all_included))

    # N1/N3: используем send_digest если кнопки переданы, иначе — send_text
    sent_urls: List[str] = []
    if item_buttons_per_msg is not None:
        # send_digest: возвращает 3 значения, берём только 2
        sent_count, sent_urls, _ = _send_digest_with_targets(
            delivery, groups, item_buttons_per_msg
        )
    else:
        # Обратная совместимость: send_text (old pipeline / test calls).
        # MAJOR 2-фикс: per-message try/except — исключение на одном сообщении
        # НЕ прерывает цикл отправки остальных. Остальные сообщения дайджеста
        # уходят; битый item отмечается на уровне pipeline (по returned failed_urls).
        for i, (msg, group_items) in enumerate(groups, 1):
            if not msg.strip():
                continue
            if len(msg) > _TG_MESSAGE_MAX:
                logger.warning(
                    "Дайджест msg[%d]: %d символов (>%d лимит)",
                    i, len(msg), _TG_MESSAGE_MAX,
                )
            try:
                ok = delivery.send_text(msg)
            except Exception as e:
                logger.warning(
                    "Дайджест msg[%d]: исключение при отправке: %s — "
                    "%d items исключены, остальные отправляются",
                    i, e, len(group_items),
                )
                continue
            if ok:
                sent_urls.extend(item.url for item in group_items)
            else:
                logger.warning(
                    "Дайджест msg[%d]: send_text=False — "
                    "%d items остаются deferred",
                    i, sum(len(it) for _, it in groups[i - 1:]),
                )
                # continue: остальные сообщения дайджеста должны уйти
                # (at-least-once сохраняется)
        sent_count = len(sent_urls)

    if not sent_urls:
        return 0, [], all_failed_urls

    try:
        marked = storage.set_digested(sent_urls)
        logger.info("Дайджест: %d items → 'digested'", marked)
    except Exception as e:
        logger.error("Дайджест: set_digested упал: %s", e)
        # Данные отправлены, но не помечены → допустимый at-least-once.
        # Следующий дайджест может продублировать, но не потерять.

    return sent_count, sent_urls, all_failed_urls