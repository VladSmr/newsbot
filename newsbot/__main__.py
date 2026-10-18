"""
Точка входа: python -m newsbot

Важно: одновременно два getUpdates-потребителя (poller и --check-telegram)
запускать НЕЛЬЗЯ — будет 409 Conflict. Poller стартует только в демон-режиме
(без --once / --digest / --check-*).
"""
import argparse
import logging
import signal
import sys
import threading
import time
from typing import Optional

from .config import load_config
from .env import telegram_token, deepseek_api_key, _write_env
from .log import setup_logging
from .llm import DeepSeekClient, LLMError
from .pipeline import NewsPipeline, _parse_bool
from .telegram_io import TelegramPoller
from .version import __version__

_run = True

_BACKOFF_SECONDS = 30  # пауза при непойманном исключении в цикле

# глобальная ссылка на поллер — для graceful shutdown
_poller: Optional["TelegramPoller"] = None
# M1: event для /digest — поллер устанавливает, основной цикл проверяет
_digest_requested = threading.Event()


def _signal_handler(signum, frame):
    global _run
    print("\n[newsbot] Получен сигнал остановки, завершаем цикл...")
    _run = False
    _digest_requested.set()  # разбудить main thread из wait
    if _poller is not None:
        _poller.stop()


def main():
    parser = argparse.ArgumentParser(
        description="newsbot — персональный новостной агрегатор"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Один цикл: собрать → дедуп → доставить → выйти",
    )
    parser.add_argument(
        "--check-telegram",
        action="store_true",
        help="Проверить токен бота (getMe), показать username",
    )
    parser.add_argument(
        "--set-chat-id",
        action="store_true",
        help="Получить chat_id через getUpdates и сохранить в .env",
    )
    parser.add_argument(
        "--check-llm",
        action="store_true",
        help="Проверить LLM-конфиг: баланс и тестовый вызов chat_json",
    )
    parser.add_argument(
        "--digest",
        action="store_true",
        help="Выполнить дайджест отложенных новостей (state='deferred') и выйти",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Показать версию приложения",
    )
    args = parser.parse_args()

    # ── диагностические команды — без полного pipeline ──
    if args.version:
        print(f"newsbot {__version__}")
        sys.exit(0)

    if args.check_telegram:
        _check_telegram()
        return

    if args.set_chat_id:
        _set_chat_id()
        return

    if args.check_llm:
        _check_llm()
        return

    if args.digest:
        _run_digest_once()
        return

    config = load_config()
    setup_logging(config["logging"])
    logger = logging.getLogger("newsbot")

    logger.info("newsbot v%s — запуск", __version__)

    pipeline = NewsPipeline(config)

    if args.once:
        logger.info("Режим: --once (один цикл)")
        try:
            pipeline.run_cycle_with_delivery()
        except Exception as e:
            logger.error("--once пал с ошибкой: %s", e, exc_info=True)
            sys.exit(1)
        finally:
            pipeline.close()
        logger.info("=== завершено ===")
        return

    # бесконечный цикл
    _digest_requested.clear()  # M1: очищаем stale events от прошлых запусков
    # ── TelegramPoller: запускается ТОЛЬКО в демон-режиме ──
    # Не запускается в --once / --digest / --check-* (диагностика getUpdates несовместима).
    _maybe_start_poller(config, pipeline, logger)

    interval = config["poll_interval_minutes"] * 60
    logger.info(
        "Режим: бесконечный цикл, интервал %d мин.",
        config["poll_interval_minutes"],
    )

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    while _run:
        try:
            # M1: обработка /digest из poller — ДО основного цикла
            if _digest_requested.is_set():
                _digest_requested.clear()
                logger.info("/digest: выполняем по запросу от poller...")
                # Minor #2: проверяем паузу перед force-digest.
                # Безопасный getattr — _is_paused может быть приватным,
                # а в отдельных окружениях pipeline может не иметь storage.
                try:
                    is_paused = getattr(pipeline, "_is_paused", lambda: False)()
                except Exception:
                    is_paused = False
                if is_paused:
                    logger.info("/digest: пропущен — bot на паузе (paused=1)")
                else:
                    try:
                        count = pipeline.run_digest(force=True)
                        if count > 0:
                            logger.info("/digest: отправлено %d новостей", count)
                        else:
                            logger.info("/digest: ничего не отправлено")
                    except Exception as e:
                        logger.error("/digest: ошибка: %s", e, exc_info=True)

            pipeline.run_cycle_with_delivery()
        except Exception as e:
            # непойманное исключение — логируем, ждём, продолжаем
            logger.error(
                "Непойманное исключение в цикле (демон жив, backoff %d с): %s",
                _BACKOFF_SECONDS, e, exc_info=True,
            )
            # ждём backoff с проверкой флага
            slept = 0
            while slept < _BACKOFF_SECONDS and _run:
                time.sleep(1)
                slept += 1
            continue

        # C3: slept инициализируется КАЖДУЮ итерацию (в т.ч. после успешного цикла)
        slept = 0
        logger.info("Пауза %d мин. до следующего цикла...",
                    config["poll_interval_minutes"])
        # M1: event.wait с таймаутом — реагирует на /digest БЫСТРО
        # (не ждёт 15 мин, просыпается через 30 с или сразу при set)
        while True:
            if not _run:
                break
            # проверяем /digest-requested даже во время паузы
            if _digest_requested.is_set():
                break
            # 30 с — батч ожидания, не слишком долго для быстрой реакции
            if _digest_requested.wait(timeout=30):
                # проснулись по set — выходим из цикла паузы
                break
            # иначе просто проснулись по таймауту — продолжаем паузу
            slept += 30
            if slept >= interval:
                break

    pipeline.close()
    if _poller is not None:
        _poller.stop()  # stop → join + finally: Storage.close() в poller thread
        _poller.close()  # только HTTP-клиент (m2: Storage уже закрыт из poller thread)
    logger.info("=== newsbot остановлен ===")


# ─────────────────────────────────────────────────────────
# TelegramPoller (Фаза 2b)
# ─────────────────────────────────────────────────────────

def _maybe_start_poller(config: dict, pipeline: NewsPipeline, logger) -> None:
    """
    Запускает TelegramPoller в демон-потоке, если:
      — telegram.token задан
      — telegram.chat_id задан

    Поллер работает НЕЗАВИСИМО от feedback.enabled — он обрабатывает
    /stop, /continue, /digest и (если feedback.enabled=true) callbacks.
    Фидбэк-обработка гейтится внутри TelegramPoller._handle_callback_query
    по feedback_enabled (параметр).

    Если feedback.enabled=false — поллер всё равно нужен для /stop|/continue|/digest.
    """
    global _poller, _digest_requested

    tg_cfg = config.get("telegram", {})
    fb_cfg = config.get("feedback", {})
    # n3: дефолт из config (default True в DEFAULT_CONFIG)
    # Minor T4-фикс: парсим через _parse_bool — bool("false")==True
    feedback_enabled = _parse_bool(fb_cfg.get("enabled", True))

    token = tg_cfg.get("token", "") or ""
    chat_id = tg_cfg.get("chat_id", "") or ""

    if not token:
        logger.warning("TelegramPoller: telegram.token not set — skipping")
        return
    if not chat_id:
        logger.warning("TelegramPoller: telegram.chat_id not set — skipping")
        return

    # M1: db_path для создания отдельного Storage в poller thread (C1)
    db_path = pipeline.storage.db_path

    poll_timeout = int(tg_cfg.get("poll_timeout", 25))

    # M1: on_digest только СИГНАЛИЗИРУЕТ — не выполняет в poller thread
    def _signal_digest():
        logger.info("/digest: poller received command, setting event for main thread")
        _digest_requested.set()

    _poller = TelegramPoller(
        token=token,
        chat_id=chat_id,
        db_path=db_path,
        on_digest=_signal_digest,
        logger=logger,
        poll_timeout=poll_timeout,
        feedback_enabled=feedback_enabled,
    )
    _poller.start()


# ─────────────────────────────────────────────────────────
# CLI-команды диагностики и привязки
# ─────────────────────────────────────────────────────────

def _mask_key(key: str) -> str:
    """
    Маскирует секретный ключ/токен.
    T2-7-фикс: показываем длину + последние 4 символа,
    НЕ первые 6 (чтобы было сложнее восстановить).
    Формат: "...<last4> (len=<N>)"
    """
    if not key:
        return "(empty)"
    n = len(key)
    last4 = key[-4:] if n >= 4 else key
    return "...{last4} (len={n})".format(last4=last4, n=n)


def _check_telegram() -> None:
    """Вызывает getMe — проверяет валидность токена."""
    import httpx
    token = telegram_token()
    if not token:
        print("ОШИБКА: TELEGRAM_BOT_TOKEN не найден в .env")
        return

    # T2-7: маскируем — длина + последние 4, не первые 6
    print(f"Токен: {_mask_key(token)}")

    try:
        resp = httpx.get(
            f"https://api.telegram.org/bot{token}/getMe",
            timeout=10.0,
        )
        data = resp.json()
        if data.get("ok"):
            result = data["result"]
            print(f"✅ Бот доступен!")
            print(f"   username: @{result.get('username', '?')}")
            print(f"   first_name: {result.get('first_name', '?')}")
            print(f"   bot_user_id: {result.get('id', '?')}")
        else:
            print(f"❌ getMe вернул ok=false: {data.get('description', resp.text)}")
    except httpx.TimeoutException:
        print("❌ Таймаут при обращении к api.telegram.org")
    except httpx.ConnectError:
        print("❌ Нет соединения: ConnectError")
    except Exception as e:
        # N4: печатаем только тип, не str(e) — может содержать URL с токеном
        print(f"❌ Ошибка: {type(e).__name__}")


def _set_chat_id() -> None:
    """
    Читает getUpdates, берёт последний message.chat.id (по update_id),
    сохраняет TELEGRAM_CHAT_ID в .env.

    M2-исправление: берёт message с наибольшим update_id,
    не первый из возвращённых. limit=100 чтобы охватить очередь,
    allowed_updates=['message'] чтобы не получать callback/edited/etc.
    """
    import httpx
    import json as _json

    token = telegram_token()
    if not token:
        print("ОШИБКА: TELEGRAM_BOT_TOKEN не найден в .env")
        return

    print("Ожидание сообщения от владельца...")
    print("  1) Откройте чат с ботом в Telegram")
    print("  2) Нажмите /start")
    print("  3) Отправьте боту любое сообщение (например, 'Привет')")
    print("  4) Дождитесь завершения скрипта")
    print()

    try:
        resp = httpx.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            params={"timeout": 30, "limit": 100,
                    "allowed_updates": _json.dumps(["message"])},
            timeout=35.0,
        )
        data = resp.json()
    except httpx.TimeoutException:
        print("Таймаут. Убедитесь, что боту отправлено сообщение.")
        return
    except httpx.ConnectError:
        print("Нет соединения: ConnectError")
        return
    except Exception as e:
        # N4: печатаем только тип, не str(e)
        print(f"Ошибка: {type(e).__name__}")
        return

    if not data.get("ok"):
        print(f"getUpdates вернул ok=false: {data.get('description')}")
        return

    updates = data.get("result", [])
    if not updates:
        print("Нет обновлений. Убедитесь, что отправили сообщение боту.")
        return

    # M2-фикс: берём update с наибольшим update_id (последнее сообщение)
    latest = max(updates, key=lambda u: u.get("update_id", 0))
    msg = latest.get("message", {})
    chat = msg.get("chat", {})
    chat_id = chat.get("id")

    if not chat_id:
        print("chat_id не найден в ответе.")
        return

    print(f"Найден chat_id: {chat_id}")
    print(f"  chat_type: {chat.get('type', '?')}")
    print(f"  first_name: {chat.get('first_name', '?')}")

    _write_env("TELEGRAM_CHAT_ID", str(chat_id))
    print()
    print(f"TELEGRAM_CHAT_ID записан в .env")
    print("Запустите --once для проверки отправки.")


def _run_digest_once() -> None:
    """
    Выполняет дайджест один раз (--digest) и выходит.
    Инициализирует pipeline, запускает run_digest(force=True), закрывает.
    Exit code 0 — всегда успех (даже если deferred пуст или ошибка log-only).

    Minor: при paused=1 CLI --digest тоже пропускает отправку и логирует
    (консистентность с daemon-путём и /digest в чате).
    """
    config = load_config()
    setup_logging(config["logging"])
    logger = logging.getLogger("newsbot")

    logger.info("=== Дайджест по запросу (--digest) ===")
    pipeline = NewsPipeline(config)
    try:
        # Minor #2: проверяем паузу — ручной CLI --digest уважает паузу
        # (консистентно с /digest в чате и daemon-run_cycle).
        try:
            is_paused = pipeline._is_paused()
        except Exception:
            is_paused = False
        if is_paused:
            logger.info("Дайджест (--digest): пропущен — bot на паузе (paused=1)")
        else:
            count = pipeline.run_digest(force=True)
            if count > 0:
                logger.info("Дайджест: отправлено %d новостей", count)
            else:
                logger.info("Дайджест: ничего не отправлено (нет отложенных новостей)")
    except Exception as e:
        logger.error("Дайджест: ошибка: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        pipeline.close()
    logger.info("=== Дайджест завершён ===")


def _check_llm() -> None:
    """
    Проверяет LLM-конфиг:
      1) Показывает модель и баланс.
      2) Делает тестовый chat_json-запрос и парсит JSON-ответ.
    Ключ не выводится целиком.
    """
    api_key = deepseek_api_key()
    if not api_key:
        print("ОШИБКА: DEEPSEEK_API_KEY не найден в .env")
        return

    config = load_config()
    llm_cfg = config.get("llm", {})

    model = llm_cfg.get("model", "")
    base_url = llm_cfg.get("base_url", "https://api.deepseek.com")

    # T2-7: маскируем — длина + последние 4
    print(f"Модель : {model}")
    print(f"API key: {_mask_key(api_key)}")
    print(f"Base URL: {base_url}")
    print()

    client = DeepSeekClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=float(llm_cfg.get("timeout_sec", 60)),
        max_retries=int(llm_cfg.get("max_retries", 3)),
    )

    # ── 1) Balance ──────────────────────────────────────────────────
    print(">>> Запрос баланса...")
    try:
        balance = client.get_balance()
        print(f"    is_available : {balance.is_available}")
        print(f"    currency      : {balance.currency!r}")
        print(f"    total_balance : {balance.total_balance}")
    except Exception as e:
        print(f"    [FAIL] Ошибка при проверке баланса: {e}")

    print()

    # ── 2) Тестовый chat_json ───────────────────────────────────────
    print(">>> Тестовый вызов chat_json...")
    test_prompt = (
        'Верни строго JSON: {"score":5,"summary":"тестовая выжимка","reason":"test"}'
    )
    try:
        result = client.chat_json(
            system="Ты ассистент, возвращающий строго JSON.",
            user=test_prompt,
        )
        print(f"    [OK] JSON распарсен успешно: {result}")
        score = result.get("score")
        summary = result.get("summary", "")
        reason = result.get("reason", "")
        if score is not None:
            print(f"    score   : {score}")
        if summary:
            print(f"    summary : {summary!r}")
        if reason:
            print(f"    reason  : {reason!r}")
    except LLMError as e:
        err_str = str(e)
        # K6: 4xx / unknown model — явное предупреждение
        is_model_error = any(kw in err_str for kw in ("404", "400", "model", "unknown"))
        if is_model_error:
            print(f"    [FAIL/WARNING] ⚠️  Модель '{model}' не распознана API!")
            print(f"    LLM-фильтр НЕ БУДЕТ РАБОТАТЬ — будет fallback (все новости заливаются).")
            print(f"    Проверьте model в config.yaml и перезапустите --check-llm.")
            print(f"    LLMError: {e}")
        else:
            print(f"    [FAIL] LLMError: {e}")
    except Exception as e:
        print(f"    [FAIL] Неожиданная ошибка: {e}")


if __name__ == "__main__":
    main()