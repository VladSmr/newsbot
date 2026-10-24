"""
TelegramDelivery — реальная отправка карточек в чат владельца через Bot API.
"""
import html
import logging
import re
import threading
import time
import urllib.parse
from typing import List, Optional, Tuple

import httpx

from ..storage import NewsItem
from . import format_time_utc

# Minor #5-фикс: strip control-chars из исходящего текста (RSS может содержать
# \x00-\x08\x0b\x0c\x0e-\x1f\x7f\x85\u2028\u2029, которые Telegram API отклоняет
# как 400 "can't parse entities"). \n, \t, \r сохраняются (имеют смысл в тексте).
_OUTGOING_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x85\u2028\u2029]")

logger = logging.getLogger("newsbot.delivery.telegram")

# Лимиты Telegram Bot API
_TG_MESSAGE_MAX = 4096
_TG_URL_MAX = 1024   # Telegram inline-кнопка URL ≤ 1024

# Telegram меряет длину текста в UTF-16 code units, а не в code points.
# Для BMP-символов 1 code point = 1 unit, для не-BMP (эмодзи и т.п.) — 2 units.
# Поэтому все проверки «влезает ли в лимит» считаем в UTF-16 units.
_UTF16_BYTES_PER_UNIT = 2

# Минимально возможная длина одного экранированного символа — &quot; (6 симв.).
# Нужна как нижняя граница usable, иначе жёсткий срез не сможет отдать
# даже один символ и зациклится/превысит лимит.
_MIN_ESCAPED_CHAR_UNITS = 6


def _utf16_units(s: str) -> int:
    """Длина строки в UTF-16 code units (как её считает Telegram)."""
    return len(s.encode("utf-16-le")) // _UTF16_BYTES_PER_UNIT

# ── Pacing: защита от 429 Too Many Requests ───────────────────────────────────
# Пауза между исходящими сообщениями (Telegram rate limit ~30 msg/sec).
# 0.6 с — достаточно для заполнения burst bucket'а без превышения лимита.
SEND_PACING_SEC = 0.6

# ── 429: границы ожидания внутри ОДНОГО вызова API ────────────────────────────
# Суммарный дедлайн ожидания 429 в рамках одного _api_call. Telegram может
# вернуть retry_after на часы — ждать такое в цикле доставки нельзя
# (Docker stop_grace_period 45 с → SIGKILL). По исчерпании дедлайна —
# TelegramDeliveryError, item остаётся pending (at-least-once).
RETRY_429_TOTAL_DEADLINE_SEC = 600
# Максимум подряд идущих 429 в одном вызове (защита от бесконечного цикла,
# если Telegram отдаёт retry_after=0/маленький, но не пускает).
MAX_CONSECUTIVE_429 = 10

# Обычные (не 429) ретраи: всего попыток и паузы между ними.
_RETRY_MAX_ATTEMPTS = 3
_RETRY_HTTP_BACKOFF_SEC = 3
_RETRY_5XX_BACKOFF_SEC = 5

# ── URL guard (общий helper для poller и digest) ──────────────────────────────

# module-level: компилируется один раз, не на каждый вызов
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def is_valid_http_url(url: str) -> str:
    """
    Проверяет URL на валидность для Telegram-кнопки / дайджест-ссылки.

    Возвращает URL как есть если валиден, или пустую строку если нет.

    Правила:
      — схема http:// или https:// (urllib.parse проверяет netloc≠'')
      — нет control-символов (\\x00-\\x1f, \\x7f-\\x9f)
      — нет whitespace / внутренних пробелов
      — длина ≤ _TG_URL_MAX
      — нет leading/trailing whitespace (reject полностью, не очищает)
    """
    if not url or url != url.strip():
        return ""

    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return ""

    # urllib.parse разбирает "https://ex.com/a b" → netloc='ex.com', path='/a b'
    # Но host будет пустым только при невалидной структуре.
    # Проверяем: схема http/https + netloc непустой + нет пробелов в path
    if not parsed.netloc:
        return ""

    # Внутренний пробел в path или query — отвергаем
    if " " in url or any(c.isspace() for c in url):
        return ""

    if _CONTROL_RE.search(url):
        return ""

    if len(url) > _TG_URL_MAX:
        return ""

    return url


def _strip_control_chars(text: str) -> str:
    """
    Minor #5-фикс: удаляет control-chars из исходящего текста.

    RSS-фиды могут содержать нуль-символы, vertical tab, form feed и прочие
    непечатаемые символы (\\x00-\\x08, \\x0b, \\x0c, \\x0e-\\x1f, \\x7f,
    \\x85, U+2028/2029). Telegram Bot API отклоняет их как 400
    «can't parse entities». \\n, \\t, \\r сохраняются (имеют значение в тексте).
    """
    return _OUTGOING_CONTROL_RE.sub("", text)

# HTTP-статусы
_HTTP_FORBIDDEN = 403
_HTTP_TOO_MANY_REQUESTS = 429

# 5xx — транзиентные, ретраим наравне с сетевыми
_RETRYABLE_SERVER_ERRORS = range(500, 600)

# ── Исключения ──────────────────────────────────────────────────────────────

class TelegramDeliveryError(Exception):
    """Базовое исключение для ошибок отправки."""
    pass


class ChatIdNotBoundError(TelegramDeliveryError):
    """chat_id не привязан или владелец не нажал /start у бота."""
    pass


class BotBlockedError(TelegramDeliveryError):
    """Бот заблокирован пользователем (HTTP 403)."""
    pass


class PermanentDeliveryError(TelegramDeliveryError):
    """Постоянная ошибка доставки (HTTP 400/413/422): bad request / too long / can't parse.

    В отличие от транзиентных ошибок (429/5xx/сеть), эти НЕ исправятся при повторе —
    item бессмысленно повторять. Помечается failed напрямую (set_state('failed')),
    но батч продолжается (уже доставленные не откатываются).
    """
    pass


# ── Клиент с lazy init (не умирает после close) ──────────────────────────────

class _LazyClient:
    """
    Обёртка над httpx.Client. Создаёт HTTP-клиент при первом запросе.
    После close() — пересоздаёт себя автоматически при следующем запросе.
    """

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout
        self._client: httpx.Client | None = None

    def _ensure(self) -> httpx.Client:
        if self._client is None or self._client.is_closed:
            # trust_env=True: подхватывает HTTP_PROXY / HTTPS_PROXY из окружения (Docker)
            self._client = httpx.Client(timeout=self._timeout, trust_env=True)
        return self._client

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self._ensure().post(url, **kwargs)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


# ── TelegramDelivery ─────────────────────────────────────────────────────────

class TelegramDelivery:
    """
    Отправляет карточки новостей в личный чат владельца
    через Telegram Bot API (sendMessage + inline keyboard).

    Использует httpx для HTTP. parse_mode — HTML.
    Все спецсимволы экранируются через html.escape().

    send_batch возвращает кортеж (delivered_urls, failed_urls).
    При ChatIdNotBoundError / BotBlockedError — re-raise (items остаются pending).
    При TelegramDeliveryError батч прерывается: отправленное — delivered,
    остаток — pending (at-least-once).
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        shutdown_event: Optional[threading.Event] = None,
    ):
        self.token = token
        self.chat_id = chat_id
        self._base_url = f"https://api.telegram.org/bot{self.token}"
        self._http = _LazyClient(timeout=30.0)
        # Прерываемые ожидания (pacing, 429, ретраи). event.wait() возвращается
        # сразу при SIGTERM/SIGINT — в отличие от time.sleep().
        self._shutdown_event = shutdown_event

    # ── ожидания ─────────────────────────────────────────────────────────────

    def _wait(self, seconds: float) -> None:
        """
        Пауза длиной seconds, прерываемая shutdown_event (если он задан).

        shutdown_event=None → обычный time.sleep (обратная совместимость).
        """
        if seconds <= 0:
            return
        if self._shutdown_event is not None:
            self._shutdown_event.wait(seconds)
        else:
            time.sleep(seconds)

    def _is_shutdown(self) -> bool:
        """True — запрошен graceful shutdown (event задан и установлен).

        getattr: подклассы/моки могут не вызывать __init__ вовсе.
        """
        event = getattr(self, "_shutdown_event", None)
        return event is not None and event.is_set()


    # ── public API ───────────────────────────────────────────────────────────

    def send(self, items: List[NewsItem]) -> Tuple[List[str], List[str]]:
        """Сингл-отправка одной новости. Возвращает (successful_urls, permanent_failed_urls)."""
        return self.send_batch(items)

    def send_batch(self, items: List[NewsItem]) -> Tuple[List[str], List[str]]:
        """
        Отправляет батч items.

        Returns:
            Tuple of (successful_urls, permanent_failed_urls).
            successful_urls — URL items, успешно доставленных (state → 'delivered').
            permanent_failed_urls — URL items с PermanentDeliveryError
            (state → 'failed', без расхода retry_count).
            При ChatIdNotBoundError / BotBlockedError — re-raise (items остаются pending).
            При TelegramDeliveryError — батч прерывается: уже отправленные
            возвращаются как successful_urls, остаток — pending (at-least-once).
        """
        if not items:
            return ([], [])

        if not self.chat_id:
            raise ChatIdNotBoundError(
                "chat_id не задан. Нажмите /start у бота, "
                "затем: python -m newsbot --set-chat-id"
            )

        successful_urls: List[str] = []
        permanent_failed_urls: List[str] = []

        for i, item in enumerate(items):
            # MINOR-фикс: при установленном shutdown_event паузы в _wait
            # возвращаются мгновенно, и цикл превращался в burst. Останавливаемся
            # ДО следующего HTTP-вызова: остаток остаётся pending (at-least-once).
            if self._is_shutdown():
                logger.warning(
                    "TelegramDelivery: получен shutdown — прекращаем отправку, "
                    "остаток (%d item) остаётся pending",
                    len(items) - i,
                )
                break

            try:
                self._send_one(item)
                successful_urls.append(item.url)
            except ChatIdNotBoundError:
                logger.warning(
                    "TelegramDelivery: chat_id не привязан. "
                    "Пропускаем оставшиеся items."
                )
                raise          # не возвращаем partial — всё останется pending
            except BotBlockedError:
                logger.error(
                    "TelegramDelivery: бот заблокирован (HTTP 403). "
                    "Оставшиеся items — pending."
                )
                raise          # аналогично
            except PermanentDeliveryError as e:
                # FIX-1: 400/413/422 — permanent: item никогда не доставится.
                # permanent_failed_urls → pipeline делает set_state('failed'),
                # НЕ попадает в retry_count. Батч продолжается (уже отправленные
                # не откатываются). pacing применяется к следующему item.
                logger.warning(
                    "TelegramDelivery: permanent сбой '%s': %s — "
                    "item → failed (permanent)",
                    item.title[:60], e,
                )
                permanent_failed_urls.append(item.url)
                # FIX-5-3: pacing применяется и после permanent-failed (continue),
                # чтобы следующий send не пошёл без задержки.
                # (pacing в конце тела цикла, без continue.)
            except TelegramDeliveryError as e:
                # сетевая/API-ошибка (в т.ч. исчерпан дедлайн 429).
                # Не долбим API остатком батча: он тоже упрётся в тот же лимит.
                # Отправленное — delivered, остаток — pending (at-least-once).
                logger.warning(
                    "TelegramDelivery: transient сбой '%s': %s — "
                    "прерываем батч, остаток (%d item) остаётся pending",
                    item.title[:60], e, len(items) - i,
                )
                break

            # pacing: пауза между сообщениями (кроме последнего), прерываемая
            if i < len(items) - 1:
                self._wait(SEND_PACING_SEC)

        return (successful_urls, permanent_failed_urls)

    def close(self) -> None:
        """Закрывает HTTP-клиент. Может быть вызван повторно без последствий."""
        self._http.close()

    def send_digest(self, text: str, item_buttons: List[dict]) -> bool:
        """
        Отправляет дайджест с инлайн-кнопками 👍/👎 на каждую новость.

        Каждая строка inline_keyboard содержит только [👍] [👎].
        Ссылка на новость — в тексте (footer <a href> в digest.py),
        НЕ дублируется кнопкой.
        callback_data = "fb:{target_id}:u" / "fb:{target_id}:d".

        item_buttons — List[dict] с полями:
          url: str — ссылка на новость (игнорируется, URL уже в тексте)
          target_id: int — feedback_target id

        При отсутствии target_id строка пропускается (без кнопок).

        Лимиты:
          — не более max_items_per_message строк (управляется на уровне pipeline)
          — текст ≤ 4096 символов (на уровне build_digest_messages)

        Returns True при успехе, False при ошибке.
        """
        if not self.chat_id:
            logger.warning("send_digest: chat_id не задан")
            return False

        rows = []
        for btn in item_buttons:
            target_id = btn.get("target_id")

            # 👍/👎-кнопки если есть target_id
            # NOTE: URL-кнопка НЕ добавляется — ссылка на новость уже
            # есть в тексте дайджеста (<a href="...">🔗</a> в digest.py footer).
            if target_id is not None:
                cb_u = f"fb:{target_id}:u"
                cb_d = f"fb:{target_id}:d"
                rows.append([
                    {"text": "👍", "callback_data": cb_u},
                    {"text": "👎", "callback_data": cb_d},
                ])

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        if rows:
            payload["reply_markup"] = {"inline_keyboard": rows}

        try:
            self._api_call("sendMessage", payload)
            return True
        except Exception as e:
            logger.warning("send_digest: сбой отправки: %s", type(e).__name__)
            return False

    def send_text(self, text: str) -> bool:
        """
        Отправляет произвольное HTML-текстовое сообщение владельцу.
        CONTRACT: text — уже готовый HTML (доставщик НЕ экранирует повторно).
        Возвращает True при успехе, False при ошибке (не бросает исключений).
        """
        if not self.chat_id:
            logger.warning("send_text: chat_id не задан")
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        try:
            self._api_call("sendMessage", payload)
            return True
        except Exception as e:
            # В10: не логируем str(e) — содержит URL с токеном
            logger.warning("send_text: сбой отправки: %s", type(e).__name__)
            return False

    # ── отправка одного item ─────────────────────────────────────────────────

    def _send_one(self, item: NewsItem) -> None:
        """Формирует и отправляет одно сообщение (и его части)."""
        parts = self._build_message_parts(item)
        for part in parts:
            self._api_call("sendMessage", part)

    # ── построение сообщения ────────────────────────────────────────────────

    def _build_message_parts(self, item: NewsItem) -> List[dict]:
        """
        Возвращает список dict'ов (по одному на каждую часть),
        готовых для POST sendMessage.

        Каждая часть ≤ _TG_MESSAGE_MAX.
        Формат: title + body + footer (+ inline-кнопка только в последней части).

        N2/P1-фикс: truncate и split — ДО html.escape, чтобы избежать обрыва
        сущностей (&am → &amp или &l → &lt→&amp;lt). Порядок:
        truncate → (fit-check по escaped) → split по СЫРОМУ → escape чанков.
        """
        # ── N2: truncate ДО html.escape (poison-message prevention) ──
        _MAX_RAW_TITLE = 250  # до escape, чтобы избежать обрыва сущностей
        _MAX_RAW_FOOTER_SOURCE = 100

        raw_title = _strip_control_chars(item.llm_title or item.title or "—")
        if len(raw_title) > _MAX_RAW_TITLE:
            raw_title = raw_title[:_MAX_RAW_TITLE]

        raw_source = _strip_control_chars(item.source or "—")
        if len(raw_source) > _MAX_RAW_FOOTER_SOURCE:
            raw_source = raw_source[:_MAX_RAW_FOOTER_SOURCE]

        # ── теперь escape (title не сломает entity) ──
        title = html.escape(raw_title)
        source_esc = html.escape(raw_source)

        # приоритет: llm_summary (выжимка LLM) → raw summary
        # Minor #5-фикс: strip control-chars (не \n/\t/\r — нужны для split по \n\n)
        raw_body = _strip_control_chars(
            item.llm_summary if item.llm_summary else item.summary
        )
        escaped_summary = html.escape(raw_body or "")

        ts = format_time_utc(getattr(item, "published_parsed", None), item.fetched_at)

        footer = f"📍 {source_esc} · {ts}"

        # ── URL-whitelist: только http/https, без control-символов, без whitespace ──
        # M3: защита от poison message при mailto:/tg:///битый URL.
        # Битый/не-http URL → кнопка НЕ добавляется (item доставляется текстом).
        # Использует общий helper is_valid_http_url (согласован с digest.py).
        safe_url = is_valid_http_url(item.url or "")
        has_button = bool(safe_url)

        # feedback_target_id: транзиентное поле из pipeline (Фаза 2b-1)
        fbtid = getattr(item, "feedback_target_id", None)

        # overhead считается по экранированным компонентам (гарантия ≤4096)
        overhead = len(title) + len(footer) + 4   # 4 = 2×\n + пустая строка
        part_text = f"{title}\n\n{escaped_summary}\n\n{footer}"

        # если влезает в одно сообщение — экранированный текст уже готов
        if _utf16_units(part_text) <= _TG_MESSAGE_MAX:
            payload = self._make_payload(
                title, escaped_summary, footer,
                has_button=has_button, url=safe_url, is_last=True,
                feedback_target_id=fbtid,
            )
            return [payload]

        # не влезает — передаём СЫРОЙ summary в _split_text.
        # P1-фикс: режем сырой текст, а экранируем каждый чанк отдельно
        # (иначе срез экранированного текста рвёт сущность → Telegram 400).
        return self._split_text(title, raw_body, footer, has_button, safe_url, fbtid)

    def _make_payload(self, title: str, body_esc: str, footer: str,
                      has_button: bool, url: str, is_last: bool,
                      suffix: str = "",
                      feedback_target_id: Optional[int] = None) -> dict:
        """
        Собирает dict для sendMessage.
        Все аргументы должны быть уже экранированы (html.escape) вызывающей стороной.
        Кнопка — через reply_markup (InlineKeyboardMarkup), не <a href> в тексте.

        При разбиении на части — клавиатура добавляется ТОЛЬКО на последнюю часть
        (is_last=True). На промежуточных — без кнопок. Это обеспечивает:
          (a) кнопки «под всей новостью» (не на каждом куске);
          (b) валидный 64-байтный callback_data без привязки к номеру части;
          (c) отсутствие лишних кнопок на промежуточных сообщениях.
        """
        text = f"{title}{suffix}\n\n{body_esc}\n\n{footer}"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
        }
        if is_last:
            rows = []
            # Строка 1: кнопка-ссылка на источник (если URL есть)
            if has_button and url:
                rows.append([{"text": "🔗 Открыть источник", "url": url}])
            # Строка 2: кнопки 👍/👎 (если есть feedback_target_id)
            if feedback_target_id is not None:
                cb_u = f"fb:{feedback_target_id}:u"
                cb_d = f"fb:{feedback_target_id}:d"
                rows.append([
                    {"text": "👍 Интересно", "callback_data": cb_u},
                    {"text": "👎 Не интересно", "callback_data": cb_d},
                ])
            if rows:
                payload["reply_markup"] = {"inline_keyboard": rows}
        return payload

    def _split_text(self, title_esc: str, body_raw: str,
                    footer_esc: str, has_button: bool, url: str,
                    feedback_target_id: Optional[int] = None) -> List[dict]:
        """
        Разбивает СЫРОЙ (неэкранированный) body на части <= _TG_MESSAGE_MAX.
        title_esc/footer_esc уже прошли html.escape() в вызывающей стороне,
        body_raw — НЕТ: каждый чанк экранируется отдельно.

        P1-фикс: раньше резался УЖЕ экранированный body, поэтому жёсткий срез
        мог попасть внутрь сущности (&quot; / &#x27; / &amp;) → Telegram 400
        «can't parse entities» → item уходил в failed после 8 ретраев (потеря
        новости). Теперь границы чанков считаются по длине html.escape(чанк),
        а срез всегда делается по СЫРОМУ тексту.

        Greedy: каждая часть наполняется пока влезает, затем следующая.
        Суффикс "(i/n)" и кнопка (только в последней) учтены в overhead.

        Клавиатура (URL + 👍/👎) добавляется ТОЛЬКО на последнюю часть.

        C2-фикс: title/source усекаются ДО html.escape() в _build_message_parts
        (_MAX_RAW_TITLE=250, _MAX_RAW_FOOTER_SOURCE=100), поэтому здесь
        title_esc/footer_esc УЖЕ ограничены и не нуждаются в повторном усечении.
        """
        result: List[dict] = []

        # overhead считается по экранированным title/footer + разделители.
        # 4 = 2×"\n" (после title и после body) + пустая строка.
        # Суффикс "(i/n)" считается ДИНАМИЧЕСКИ: его длина зависит от числа
        # частей, а при ≥100 частях старая константа "+12" не хватала.
        base_overhead = _utf16_units(title_esc) + _utf16_units(footer_esc) + 4

        def _pack(suffix_units: int) -> List[str]:
            """Укладывает body в части с учётом резерва под суффикс."""
            usable = _TG_MESSAGE_MAX - base_overhead - suffix_units
            # B7: гарантируем usable ≥ 1 (иначе разбиение зациклится),
            # но не меньше минимальной длины экранированного символа —
            # иначе _hard_slice_raw не сможет отдать ни одного символа.
            usable = max(usable, _MIN_ESCAPED_CHAR_UNITS)

            # разбиваем СЫРОЙ body по абзацам, бьём oversized-абзацы.
            # chunks хранит УЖЕ экранированные куски (каждый — валидный HTML).
            chunks: List[str] = []
            for para in (body_raw or "").split("\n\n"):
                if not para.strip():
                    continue
                para_esc = html.escape(para)
                if _utf16_units(para_esc) <= usable:
                    chunks.append(para_esc)
                else:
                    chunks.extend(self._split_paragraph(para, usable))

            # greedy packing
            packed: List[str] = []
            buf = ""
            for chunk in chunks:
                if not buf:
                    buf = chunk
                elif (_utf16_units(buf) + 2 + _utf16_units(chunk)
                      + base_overhead + suffix_units <= _TG_MESSAGE_MAX):
                    buf += "\n\n" + chunk
                else:
                    packed.append(buf)
                    buf = chunk
            if buf:
                packed.append(buf)
            return packed

        # Итеративно уточняем резерв под суффикс: число частей зависит от
        # резерва, а резерв — от числа частей. Резерв только РАСТЁТ
        # (монотонно), поэтому цикл сходится за 1–3 итерации.
        total_hint = 1
        packed: List[str] = []
        for _ in range(10):
            suffix_units = _utf16_units(f" ({total_hint}/{total_hint})") if total_hint > 1 else 0
            packed = _pack(suffix_units)
            # Достаточно, если частей не больше, чем предполагали: тогда
            # фактический суффикс "(i/total)" не длиннее зарезервированного.
            if len(packed) <= total_hint:
                break
            total_hint = len(packed)
        else:
            # Страховка от расхождения: резервируем с запасом на 2 знака.
            packed = _pack(_utf16_units(f" ({len(packed)}/{len(packed)})") + 2)

        if not packed:
            # Minor #6-фикс: body после split оказался whitespace-only.
            # packed пуст → result пуст → final пуст → item доставлен без текста.
            # Fallback: title + footer (без body). Бросать ошибку нельзя
            # (URL уже в successful_urls → item был бы marked delivered повторно).
            logger.warning("_split_text: body whitespace-only, fallback title+footer")
            packed = [""]

        result = [(part, i == len(packed) - 1) for i, part in enumerate(packed)]

        # проставить суффиксы и построить финальные payload'ы
        total = len(result)
        final: List[dict] = []
        for i, (body_part, _) in enumerate(result):
            suffix = f" ({i+1}/{total})" if total > 1 else ""
            is_last = (i == total - 1)
            final.append(self._make_payload(
                title_esc, body_part, footer_esc,
                has_button=(has_button and is_last),
                url=(url if is_last else ""),
                is_last=is_last,
                suffix=suffix,
                feedback_target_id=(feedback_target_id if is_last else None),
            ))
        return final

    def _split_paragraph(self, para: str, max_len: int) -> List[str]:
        """
        Бьёт длинный СЫРОЙ абзац на куски, у которых html.escape(кусок) ≤ max_len.

        Режем СЫРЫЙ текст и экранируем каждый кусок ОТДЕЛЬНО — это гарантирует,
        что сущность (&quot; / &#x27; / &amp; / &lt;) никогда не окажется разорвана
        границей чанка (P1: иначе Telegram 400 «can't parse entities» и потеря
        новости после 8 ретраев).

        Приоритет границ: предложение → слово → жёсткий срез по символам
        (жёсткий срез тоже по сырому тексту).
        """
        parts: List[str] = []

        def _emit(unit: str) -> None:
            """Кладёт один юнит; если не влезает — режет по словам/символам."""
            unit = unit.strip()
            if not unit:
                return
            if _utf16_units(html.escape(unit)) <= max_len:
                parts.append(html.escape(unit))
                return
            # юнит длиннее лимита — режем по словам
            buf = ""
            for word in unit.split():
                candidate = (buf + " " + word) if buf else word
                if _utf16_units(html.escape(candidate)) <= max_len:
                    buf = candidate
                    continue
                if buf:
                    parts.append(html.escape(buf))
                    buf = ""
                if _utf16_units(html.escape(word)) <= max_len:
                    buf = word
                else:
                    parts.extend(self._hard_slice_raw(word, max_len))
            if buf:
                parts.append(html.escape(buf))

        buf = ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if not sent.strip():
                continue
            candidate = (buf + " " + sent).strip() if buf else sent
            if _utf16_units(html.escape(candidate)) <= max_len:
                buf = candidate
            else:
                _emit(buf)   # всегда безопасно: _emit режет до max_len
                buf = sent

        if buf.strip():
            _emit(buf)   # финальный буфер — тот же хелпер
        return parts

    @staticmethod
    def _hard_slice_raw(text: str, max_len: int) -> List[str]:
        """
        Жёсткий срез СЫРОГО текста на куски, каждый из которых после
        html.escape() не длиннее max_len. Возвращает УЖЕ экранированные куски.
        """
        out: List[str] = []
        rest = text
        while rest:
            lo, hi, best = 1, len(rest), 0
            while lo <= hi:
                mid = (lo + hi) // 2
                if _utf16_units(html.escape(rest[:mid])) <= max_len:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best <= 0:
                # max_len меньше длины экранированного одного символа
                # (&quot; = 6 units). Валидного среза не существует —
                # отдаём один символ, чтобы не зациклиться, и явно
                # сигнализируем аномалию в лог (usable клампится в
                # _split_text, поэтому на практике сюда не попадаем).
                logger.warning(
                    "_hard_slice_raw: max_len=%d < минимальной длины "
                    "экранированного символа — чанк неизбежно превысит лимит",
                    max_len,
                )
                best = 1
            out.append(html.escape(rest[:best]))
            rest = rest[best:]
        return out

    # ── HTTP-вызовы ──────────────────────────────────────────────────────────

    def _api_call(self, method: str, payload: dict) -> dict:
        """
        Вызывает Telegram Bot API.

        Обрабатывает:
          - все сетевые ошибки (базовый httpx.HTTPError) → retry до 3 раз
          - 429 retry_after → ждём указанное время, но не дольше суммарного
            дедлайна RETRY_429_TOTAL_DEADLINE_SEC на ОДИН вызов; 429 НЕ
            расходует обычный attempt; не более MAX_CONSECUTIVE_429 подряд.
            При исчерпании дедлайна/лимита → TelegramDeliveryError.
          - 403 → BotBlockedError
          - 5xx → retry до 3 раз
          - невалидный JSON → TelegramDeliveryError
          - прочие 4xx → TelegramDeliveryError

        Все ожидания — через self._wait (прерываемо по shutdown_event).

        В10-фикс: ловим базовый httpx.HTTPError (а не отдельные подтипы),
        чтобы UnsupportedProtocol и прочие не-catched подтипы не просочились
        с сырым str(e), содержащим URL с токеном.
        """
        attempt = 0
        count_429 = 0
        waited_429 = 0.0

        while attempt < _RETRY_MAX_ATTEMPTS:
            # MINOR-фикс: shutdown проверяется в начале КАЖДОЙ попытки —
            # включая повторы после 429/5xx/сетевых ошибок (continue возвращается
            # сюда же). Без этого паузы обнулялись, а повторы шли подряд.
            # Остаток батча остаётся pending (at-least-once).
            if self._is_shutdown():
                raise TelegramDeliveryError(
                    "shutdown: отправка прервана, item остаётся pending"
                )

            try:
                resp = self._http.post(self._base_url + "/" + method, json=payload)
            except httpx.HTTPError as e:
                # В10: ловим базовый HTTPError — все подтипы (ConnectError,
                # TimeoutException, UnsupportedProtocol, RemoteProtocolError …).
                # Сырое исключение НЕ логируем — содержит URL с токеном.
                attempt += 1
                logger.warning(
                    "Telegram API %s: HTTP-ошибка (%s), попытка %d/%d",
                    method, type(e).__name__, attempt, _RETRY_MAX_ATTEMPTS,
                )
                if attempt < _RETRY_MAX_ATTEMPTS:
                    self._wait(_RETRY_HTTP_BACKOFF_SEC)
                    continue
                # типизированное исключение без URL
                raise TelegramDeliveryError(
                    f"HTTP-ошибка {type(e).__name__} после "
                    f"{_RETRY_MAX_ATTEMPTS} попыток"
                )

            # ── 429 — rate limit ──
            if resp.status_code == _HTTP_TOO_MANY_REQUESTS:
                try:
                    retry_after = int(resp.json().get(
                        "parameters", {}
                    ).get("retry_after", 60))
                except Exception:
                    retry_after = 60
                # Защита от 0/отрицательных значений — иначе busy-loop.
                retry_after = max(retry_after, 1)

                count_429 += 1
                if count_429 > MAX_CONSECUTIVE_429:
                    raise TelegramDeliveryError(
                        f"rate limit (429): {MAX_CONSECUTIVE_429} подряд — отказ "
                        f"от дальнейших попыток"
                    )

                remaining = RETRY_429_TOTAL_DEADLINE_SEC - waited_429
                if retry_after > remaining:
                    # Ждать бессмысленно: даже полный остаток дедлайна не
                    # покроет retry_after. item остаётся pending (at-least-once).
                    raise TelegramDeliveryError(
                        f"rate limit (429): retry_after={retry_after} с превышает "
                        f"остаток дедлайна {max(remaining, 0):.0f} с "
                        f"(лимит {RETRY_429_TOTAL_DEADLINE_SEC} с на вызов)"
                    )

                logger.warning(
                    "Telegram API %s: rate limit (429). Ждём %d с "
                    "(429 подряд: %d/%d, израсходовано %.0f/%d с)",
                    method, retry_after, count_429, MAX_CONSECUTIVE_429,
                    waited_429, RETRY_429_TOTAL_DEADLINE_SEC,
                )
                # 429 НЕ расходует attempt — ждём и повторяем тот же вызов.
                self._wait(retry_after)
                waited_429 += retry_after
                continue

            # ── 403 — бот заблокирован ──
            if resp.status_code == _HTTP_FORBIDDEN:
                raise BotBlockedError("Бот заблокирован пользователем (HTTP 403).")

            # ── 5xx — транзиентная ошибка сервера (M1) ──
            if resp.status_code in _RETRYABLE_SERVER_ERRORS:
                attempt += 1
                logger.warning(
                    "Telegram API %s: серверная ошибка %d (попытка %d/%d)",
                    method, resp.status_code, attempt, _RETRY_MAX_ATTEMPTS,
                )
                if attempt < _RETRY_MAX_ATTEMPTS:
                    self._wait(_RETRY_5XX_BACKOFF_SEC)
                    continue
                raise TelegramDeliveryError(
                    f"Серверная ошибка HTTP {resp.status_code} после "
                    f"{_RETRY_MAX_ATTEMPTS} попыток"
                )

            # ── прочие ошибки (4xx без special handling) ──
            if not resp.is_success:
                try:
                    body = resp.json()
                    desc = body.get("description", "unknown")
                except Exception:
                    desc = resp.text[:100] if resp.text else "unknown"
                # Major #4-фикс: 400/413/422 — permanent (не исправятся при retry).
                # 429 обрабатывается выше (retry-after). 403 → BotBlockedError.
                status = resp.status_code
                if status in (400, 413, 422):
                    raise PermanentDeliveryError(
                        f"HTTP {status}: {desc}"
                    )
                raise TelegramDeliveryError(
                    f"HTTP {status}: {desc}"
                )

            # ── успешный ответ — парсим JSON безопасно (M1) ──
            try:
                result = resp.json()
            except Exception as e:
                raise TelegramDeliveryError(
                    f"Невалидный JSON ответа: {e}"
                )

            if not result.get("ok"):
                raise TelegramDeliveryError(
                    f"Telegram returned ok=false: {result.get('description', '?')}"
                )

            return result.get("result", {})

        # сюда не попадаем (цикл либо return, либо raise), но для типизатора:
        raise TelegramDeliveryError(
            f"{method} не удался после {_RETRY_MAX_ATTEMPTS} попыток"
        )
