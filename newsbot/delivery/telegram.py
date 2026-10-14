"""
TelegramDelivery — реальная отправка карточек в чат владельца через Bot API.
"""
import html
import logging
import re
import time
from typing import List, Optional

import httpx

from ..storage import NewsItem
from . import format_time_utc

logger = logging.getLogger("newsbot.delivery.telegram")

# Лимиты Telegram Bot API
_TG_MESSAGE_MAX = 4096
_TG_URL_MAX = 1024   # Telegram inline-кнопка URL ≤ 1024

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

    send_batch возвращает список URL успешно отправленных items.
    При любом неотправленном item бросает TelegramDeliveryError →
    pipeline помечает delivered ТОЛЬКО вернувшиеся URL.
    """

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._base_url = f"https://api.telegram.org/bot{self.token}"
        self._http = _LazyClient(timeout=30.0)

    # ── public API ───────────────────────────────────────────────────────────

    def send(self, items: List[NewsItem]) -> List[str]:
        """Сингл-отправка одной новости. Возвращает [url] при успехе, [] при ошибке."""
        return self.send_batch(items)

    def send_batch(self, items: List[NewsItem]) -> List[str]:
        """
        Отправляет батч items.

        Возвращает список URL успешно доставленных items.
        При ChatIdNotBoundError / BotBlockedError — re-raise (items остаются pending).
        При прочих ошибках (сеть, 5xx, API) — логирует и продолжает;
        успешно отправленные URL возвращаются, остальные остаются pending
        и будут обработаны retry в следующем цикле.
        """
        if not items:
            return []

        if not self.chat_id:
            raise ChatIdNotBoundError(
                "chat_id не задан. Нажмите /start у бота, "
                "затем: python -m newsbot --set-chat-id"
            )

        successful_urls: List[str] = []

        for item in items:
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
            except TelegramDeliveryError as e:
                # сетевая/API-ошибка → логируем, продолжаем,
                # item остаётся pending, успешные — уже в successful_urls
                logger.error(
                    "TelegramDelivery: сбой отправки '%s': %s",
                    item.title[:60], e,
                )
                # не поднимаем дальше — отдаём то, что отправилось

        return successful_urls

    def close(self) -> None:
        """Закрывает HTTP-клиент. Может быть вызван повторно без последствий."""
        self._http.close()

    def send_digest(self, text: str, item_buttons: List[dict]) -> bool:
        """
        Отправляет дайджест с инлайн-кнопками на каждую новость.

        Каждая строка inline_keyboard: [URL-кнопка] [👍] [👎].
        callback_data = "fb:{target_id}:u" / "fb:{target_id}:d".

        item_buttons — List[dict] с полями:
          url: str — ссылка на новость (опционально)
          target_id: int — feedback_target id

        При отсутствии URL строка содержит только 👍/👎.
        При отсутствии feedback (target_id=None) строка содержит только URL-кнопку.
        При отсутствии обоих — строка пропускается.

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
            url = btn.get("url") or ""
            target_id = btn.get("target_id")

            # m1 (T2): URL-whitelist — кнопка только для http:// / https://
            # (аналогично whitelist в тексте дайджеста / digest.py)
            url_clean = url.strip() if url else ""
            if url_clean:
                if not (
                    url_clean.startswith("http://")
                    or url_clean.startswith("https://")
                ):
                    url_clean = ""   # невалидный URL — кнопку не добавляем
                elif len(url_clean) > _TG_URL_MAX:
                    url_clean = ""   # слишком длинный — кнопку не добавляем

            url_btn = None
            if url_clean:
                url_btn = {"text": "🔗", "url": url_clean}

            # 👍/👎-кнопки если есть target_id
            thumbs = None
            if target_id is not None:
                cb_u = f"fb:{target_id}:u"
                cb_d = f"fb:{target_id}:d"
                thumbs = [
                    {"text": "👍", "callback_data": cb_u},
                    {"text": "👎", "callback_data": cb_d},
                ]

            # Строка: всегда 1–3 кнопки; пустая строка пропускается
            row_btns = [b for b in [url_btn] if b is not None]
            if thumbs:
                row_btns.extend(thumbs)

            if row_btns:
                rows.append(row_btns)

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
        Отправляет произвольное текстовое сообщение владельцу.
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

        N2-фикс: truncate ДО html.escape, чтобы избежать обрыва сущностей
        (&am → &amp или &l → &lt→&amp;lt). Порядок: truncate → escape → split.
        """
        # ── N2: truncate ДО html.escape (poison-message prevention) ──
        _MAX_RAW_TITLE = 250  # до escape, чтобы избежать обрыва сущностей
        _MAX_RAW_FOOTER_SOURCE = 100

        raw_title = (item.title or "—")
        if len(raw_title) > _MAX_RAW_TITLE:
            raw_title = raw_title[:_MAX_RAW_TITLE]

        raw_source = item.source or "—"
        if len(raw_source) > _MAX_RAW_FOOTER_SOURCE:
            raw_source = raw_source[:_MAX_RAW_FOOTER_SOURCE]

        # ── теперь escape (title не сломает entity) ──
        title = html.escape(raw_title)
        source_esc = html.escape(raw_source)

        # приоритет: llm_summary (выжимка LLM) → raw summary
        raw_body = item.llm_summary if item.llm_summary else item.summary
        escaped_summary = html.escape(raw_body or "")

        ts = format_time_utc(getattr(item, "published_parsed", None), item.fetched_at)

        footer = f"📍 {source_esc} · {ts}"

        # inline-кнопка — вместо <a href> в тексте (правильный API-способ)
        url = item.url or ""
        has_button = bool(url and len(url) <= _TG_URL_MAX)

        # feedback_target_id: транзиентное поле из pipeline (Фаза 2b-1)
        fbtid = getattr(item, "feedback_target_id", None)

        # overhead считается по экранированным компонентам (гарантия ≤4096)
        overhead = len(title) + len(footer) + 4   # 4 = 2×\n + пустая строка
        part_text = f"{title}\n\n{escaped_summary}\n\n{footer}"

        # если влезает в одно сообщение — экранированный текст уже готов
        if len(part_text) <= _TG_MESSAGE_MAX:
            payload = self._make_payload(
                title, escaped_summary, footer,
                has_button=has_button, url=url, is_last=True,
                feedback_target_id=fbtid,
            )
            return [payload]

        # не влезает — передаём экранированный summary в _split_text
        return self._split_text(title, escaped_summary, footer, has_button, url, fbtid)

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

    def _split_text(self, title_esc: str, body_esc: str,
                    footer_esc: str, has_button: bool, url: str,
                    feedback_target_id: Optional[int] = None) -> List[dict]:
        """
        Разбивает ЭКРАНИРОВАННЫЙ body (body_esc) на части <= _TG_MESSAGE_MAX.
        Все аргументы уже прошли html.escape() в вызывающей стороне.
        Greedy: каждая часть наполняется пока влезает, затем следующая.
        Суффикс "(i/n)" и кнопка (только в последней) учтены в overhead.

        Клавиатура (URL + 👍/👎) добавляется ТОЛЬКО на последнюю часть:
          - избегаем «кнопки на каждом куске»;
          - feedback_target_id не зависит от номера части;
          - Telegram гарантирует, что callback приходит по message_id последнего
            сообщения (при отправке нескольких — Telegram использует последний).

        C2-фикс: title/source усекаются ДО html.escape() в _build_message_parts
        (_MAX_RAW_TITLE=250, _MAX_RAW_FOOTER_SOURCE=100), поэтому здесь
        экранированные title_esc/footer_esc УЖЕ ограничены и НЕ нуждаются
        в повторном усечении. Повторное усечение title_esc при наличии &/</>/"
        может обрезать сущность (&am → &amp → Telegram 400). Не усекаем.
        """
        result: List[dict] = []

        # C2: лимиты уже заданы ДО escape в _build_message_parts —
        # title/source обрезаны по байтам ДО html.escape,
        # поэтому title_esc/footer_esc гарантированно не содержат
        # незавершённых сущностей и НЕ требуют повторного усечения.
        # Оставляем их как есть.

        # overhead считается по экранированным title/footer + suffix
        overhead = len(title_esc) + len(footer_esc) + 12   # 12 ~ "(99/99)" + margin
        usable = _TG_MESSAGE_MAX - overhead
        # B7: гарантируем usable >= 1 — иначе _split_paragraph с max_len<=0 зациклится
        usable = max(usable, 1)

        # разбиваем body по абзацам, бьём oversized-абзацы
        chunks: List[str] = []
        for para in body_esc.split("\n\n"):
            if not para.strip():
                continue
            if len(para) > usable:
                chunks.extend(self._split_paragraph(para, usable))
            else:
                chunks.append(para)

        # greedy packing; result хранит (body_esc, is_last)
        buf = ""
        for chunk in chunks:
            if not buf:
                buf = chunk
            elif len(buf) + 2 + len(chunk) + len(title_esc) + len(footer_esc) + 4 + 12 <= _TG_MESSAGE_MAX:
                buf += "\n\n" + chunk
            else:
                result.append((buf, False))
                buf = chunk

        if buf:
            result.append((buf, True))   # последняя — с кнопкой

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
        Бьёт длинный абзац по предложениям.
        Если отдельное предложение длиннее max_len — жёсткий срез (chunk > max_len
        невозможен, иначе _split_text получит oversized chunk и первая часть превысит 4096).
        """
        parts: List[str] = []

        def _emit(b: str) -> None:
            """Emit chunk: slice to max_len before append, repeat until fits."""
            b = b.strip()
            while len(b) > max_len:
                parts.append(b[:max_len].rstrip())
                b = b[max_len:].lstrip()
            if b:
                parts.append(b)

        buf = ""
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if not sent.strip():
                continue
            if buf and len(buf) + len(sent) + 1 > max_len:
                _emit(buf)   # всегда безопасно: _emit режет до max_len
                buf = sent
            else:
                buf = (buf + " " + sent).strip()

        if buf.strip():
            _emit(buf)   # финальный буфер — тот же хелпер
        return parts

    # ── HTTP-вызовы ──────────────────────────────────────────────────────────

    def _api_call(self, method: str, payload: dict) -> dict:
        """
        Вызывает Telegram Bot API.

        Обрабатывает:
          - все сетевые ошибки (базовый httpx.HTTPError) → retry до 3 раз
          - 429 retry_after → ждём указанное время (capped 120 сек)
          - 403 → BotBlockedError
          - 5xx → retry до 3 раз
          - невалидный JSON → TelegramDeliveryError
          - прочие 4xx → TelegramDeliveryError

        В10-фикс: ловим базовый httpx.HTTPError (а не отдельные подтипы),
        чтобы UnsupportedProtocol и прочие не-catched подтипы не просочились
        с сырым str(e), содержащим URL с токеном.
        """
        for attempt in range(3):
            try:
                resp = self._http.post(self._base_url + "/" + method, json=payload)
            except httpx.HTTPError as e:
                # В10: ловим базовый HTTPError — все подтипы (ConnectError,
                # TimeoutException, UnsupportedProtocol, RemoteProtocolError …).
                # Сырое исключение НЕ логируем — содержит URL с токеном.
                logger.warning(
                    "Telegram API %s: HTTP-ошибка (%s), попытка %d/3",
                    method, type(e).__name__, attempt + 1,
                )
                if attempt < 2:
                    time.sleep(3)
                    continue
                # типизированное исключение без URL
                raise TelegramDeliveryError(
                    f"HTTP-ошибка {type(e).__name__} после 3 попыток"
                )

            # ── 429 — rate limit ──
            if resp.status_code == _HTTP_TOO_MANY_REQUESTS:
                try:
                    retry_after = resp.json().get(
                        "parameters", {}
                    ).get("retry_after", 60)
                except Exception:
                    retry_after = 60
                logger.warning(
                    "Telegram API %s: rate limit (429). Ждём %d с",
                    method, retry_after,
                )
                time.sleep(min(retry_after, 120))
                continue

            # ── 403 — бот заблокирован ──
            if resp.status_code == _HTTP_FORBIDDEN:
                raise BotBlockedError("Бот заблокирован пользователем (HTTP 403).")

            # ── 5xx — транзиентная ошибка сервера (M1) ──
            if resp.status_code in _RETRYABLE_SERVER_ERRORS:
                logger.warning(
                    "Telegram API %s: серверная ошибка %d (попытка %d/3)",
                    method, resp.status_code, attempt + 1,
                )
                if attempt < 2:
                    time.sleep(5)
                    continue
                raise TelegramDeliveryError(
                    f"Серверная ошибка HTTP {resp.status_code} после 3 попыток"
                )

            # ── прочие ошибки (4xx без special handling) ──
            if not resp.is_success:
                try:
                    body = resp.json()
                    desc = body.get("description", "unknown")
                except Exception:
                    desc = resp.text[:100] if resp.text else "unknown"
                raise TelegramDeliveryError(
                    f"HTTP {resp.status_code}: {desc}"
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
        raise TelegramDeliveryError(f"{method} не удался после 3 попыток")