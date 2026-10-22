"""
Точка входа: python -m newsbot

Важно: одновременно два getUpdates-потребителя (poller и --check-telegram)
запускать НЕЛЬЗЯ — будет 409 Conflict. Poller стартует только в демон-режиме
(без --once / --digest / --check-*).
"""
import argparse
import ctypes
import errno
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from .config import load_config
from .env import telegram_token, deepseek_api_key, telegram_owner_id, _write_env
from .log import setup_logging, _minimal_sanitizer_handler
from .llm import DeepSeekClient, LLMError
from .pipeline import NewsPipeline, _parse_bool
from .telegram_io import TelegramPoller
from .version import __version__

# ── graceful shutdown: threading.Event вместо bool-флага ─────────────────────
# Event работает с threading.Condition/wait(), bool — нет.
# signal.signal ставится ТОЛЬКО в main thread.
_shutdown_event = threading.Event()

_BACKOFF_SECONDS = 30  # пауза при непойманном исключении в цикле

# Бюджет ожидания TelegramPoller при graceful shutdown (сек).
# stop() + close() вместе НЕ должны превышать этот бюджет: stop() делает
# первый join (poll_timeout+5), close() получает лишь остаток дедлайна.
# Должно быть < stop_grace_period (45s в docker-compose.yml), чтобы успеть
# выполнить pipeline.close() и освободить lock-файл до SIGKILL.
_POLLER_SHUTDOWN_BUDGET_SEC = 40.0

# глобальная ссылка на поллер — для graceful shutdown
_poller: Optional["TelegramPoller"] = None
# M1: event для /digest — поллер устанавливает, основной цикл проверяет
_digest_requested = threading.Event()

# ── single-instance lock (демон) ────────────────────────────────────────────
# Lock-файл: <db_dir>/newsbot.lock. Захватывается при демон-старте,
# освобождается в finally при аккуратном выходе.
# Для --once/--digest/--check-* блокировка НЕ используется (one-shot, быстрые).
#
# MAJOR-фикс (lock-liveness): раньше «живость» определялась ТОЛЬКО по PID из
# файла. После SIGKILL/OOM/reboot в файле оставался PID (например 7 — типичный
# PID приложения под tini), новый контейнер в своём PID-namespace получал ТОТ ЖЕ
# PID, os.kill(pid, 0) отвечал «жив» (это он сам) → «Уже запущен» → exit(1) →
# restart: unless-stopped → вечный рестарт-луп, лечился только ручным rm.
# Теперь основной механизм — эксклюзивная БЛОКИРОВКА ФАЙЛА (ядро снимает её
# при смерти процесса):
#   POSIX   — fcntl.flock(LOCK_EX | LOCK_NB);
#   Windows — msvcrt.locking(LK_NBLCK) на байте вне данных файла.
# Семантика «уже запущен» = не удалось взять блокировку (а не «PID жив»).
# Lock-файл при этом может оставаться на диске — это нормально.
# PID-эвристика (_acquire_lock_pid_fallback) используется ТОЛЬКО как fallback,
# если механизм блокировки недоступен (экзотическая ФС / нет fcntl и msvcrt).
_LOCK_FILE: Optional["Path"] = None
# Держим fd открытым всё время жизни демона — закрытие снимает блокировку.
_LOCK_FD: Optional[int] = None
# Windows/msvcrt: блокируем байт по смещению 4096 (вне содержимого с PID).
# Блокировать сам байт 0 нельзя: Windows запрещает запись в заблокированный
# диапазон даже владельцу блокировки — PID не записать.
_LOCK_WIN_OFFSET = 4096


def _lock_fd_exclusive(fd: int) -> Optional[bool]:
    """
    Пытается взять эксклюзивную блокировку файла по открытому fd.

    Returns:
      True  — блокировка взята (мы владелец);
      False — блокировку держит другой процесс (живой владелец);
      None  — механизм недоступен → вызывающий откатывается на PID-эвристику.
    """
    try:
        import fcntl  # POSIX-only
    except ImportError:
        fcntl = None  # type: ignore[assignment]

    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EAGAIN):
                return False
            return None

    try:
        import msvcrt  # Windows-only
    except ImportError:
        return None

    try:
        os.lseek(fd, _LOCK_WIN_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        return True
    except OSError as e:
        if e.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            return False
        return None


def _unlock_fd(fd: int) -> None:
    """Снимает блокировку, взятую _lock_fd_exclusive (best-effort)."""
    try:
        import fcntl
    except ImportError:
        fcntl = None  # type: ignore[assignment]

    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        return

    try:
        import msvcrt
    except ImportError:
        return
    try:
        os.lseek(fd, _LOCK_WIN_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


def _write_pid_to_fd(fd: int, pid: int) -> None:
    """Перезаписывает содержимое lock-файла нашим PID (best-effort)."""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, str(pid).encode("ascii"))
    except OSError:
        pass


def _read_lock_pid(lock_path: "Path") -> Optional[int]:
    """PID из lock-файла; None если файла нет / содержимое не число."""
    try:
        return int(lock_path.read_text().strip())
    except (ValueError, OSError):
        return None


def _pid_blocks_start(pid: Optional[int], my_pid: int) -> bool:
    """
    True если PID из lock-файла указывает на ЖИВОЙ чужой процесс.

    Собственный PID заведомо stale: мы только что стартовали, второго
    экземпляра в этом же процессе быть не может. Это же закрывает сценарий
    PID-reuse (контейнер получил тот же PID после OOM/reboot).
    Используется только в PID-fallback (когда блокировка файла недоступна).
    """
    if pid is None or pid == my_pid:
        return False
    return _is_process_alive(pid)


def _acquire_lock(lock_path: "Path", my_pid: int) -> bool:
    """
    Захватывает single-instance lock.

    Основной путь — эксклюзивная блокировка файла (flock/msvcrt): ядро снимает
    её при смерти процесса, поэтому stale-файл с «живым» PID больше не мешает
    старту (Major-фикс: вечный рестарт-луп после SIGKILL/OOM/reboot).
    Побочный эффект: TOCTOU-окна с unlink чужого живого lock нет — файл не
    удаляется вовсе, «занято» определяется неудачей блокировки.

    Fallback (блокировка недоступна) — прежняя PID-эвристика с атомарной
    публикацией через os.link() и защитой от собственного PID.

    Returns True если lock получен; при занятом lock — sys.exit(1).
    """
    global _LOCK_FD

    fd: Optional[int] = None
    try:
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        fd = None

    if fd is not None:
        try:
            got = _lock_fd_exclusive(fd)
        except Exception:
            got = None

        if got is True:
            _write_pid_to_fd(fd, my_pid)
            _LOCK_FD = fd
            return True

        if got is False:
            # Блокировку держит реальный живой владелец.
            pid = _read_lock_pid(lock_path)
            if pid is None:
                # Владелец мог ещё не дописать PID — короткий повтор.
                for _ in range(5):
                    time.sleep(0.1)
                    pid = _read_lock_pid(lock_path)
                    if pid is not None:
                        break
            if pid is not None:
                print(
                    f"[newsbot] Уже запущен (pid {pid}). Выход.",
                    file=sys.stderr,
                )
            else:
                print("[newsbot] Уже запущен (lock занят). Выход.", file=sys.stderr)
            try:
                os.close(fd)
            except OSError:
                pass
            sys.exit(1)

        # got is None → механизм недоступен, уходим в PID-fallback.
        try:
            os.close(fd)
        except OSError:
            pass

    return _acquire_lock_pid_fallback(lock_path, my_pid)


def _acquire_lock_pid_fallback(lock_path: "Path", my_pid: int) -> bool:
    """
    Fallback-захват lock по PID-эвристике (если блокировка файла недоступна).

    Атомарность публикации: PID пишется в temp-файл в той же директории и
    публикуется через os.link() (атомарный create-if-not-exists) — lock-файл
    появляется СРАЗУ с содержимым. На ФС без hardlink — O_CREAT|O_EXCL.

    Отличие от прежней версии (Major, минимум): PID == собственный считается
    stale, а перед unlink PID перечитывается (защита от TOCTOU).

    Returns True если lock получен; при занятом lock — sys.exit(1).
    """

    def _publish() -> bool:
        """Публикует lock с PID. False — lock уже существует."""
        tmp_path = lock_path.with_name(f"{lock_path.name}.{my_pid}.tmp")
        try:
            tmp_path.write_text(str(my_pid), encoding="ascii")
            try:
                os.link(str(tmp_path), str(lock_path))
            except FileExistsError:
                return False
            except OSError:
                # ФС без поддержки hardlink → эксклюзивное создание.
                try:
                    fd = os.open(
                        str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
                    )
                except FileExistsError:
                    return False
                try:
                    os.write(fd, str(my_pid).encode())
                finally:
                    os.close(fd)
            return True
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

    if _publish():
        return True

    pid = _read_lock_pid(lock_path)

    # None = пустой/нечитаемый файл. Даём короткий шанс: возможно, другой
    # процесс ещё дописывает содержимое.
    if pid is None:
        for _ in range(5):
            time.sleep(0.1)
            pid = _read_lock_pid(lock_path)
            if pid is not None:
                break

    if _pid_blocks_start(pid, my_pid):
        print(f"[newsbot] Уже запущен (pid {pid}). Выход.", file=sys.stderr)
        sys.exit(1)

    # TOCTOU: перечитываем PID непосредственно перед удалением — если владелец
    # сменился на живой чужой, чужой lock не трогаем.
    current = _read_lock_pid(lock_path)
    if current != pid and _pid_blocks_start(current, my_pid):
        print(f"[newsbot] Уже запущен (pid {current}). Выход.", file=sys.stderr)
        sys.exit(1)

    # Не живой (stale) → удаляем и пробуем снова.
    try:
        lock_path.unlink()
    except OSError:
        pass

    if _publish():
        return True

    print("[newsbot] Lock занят другим процессом. Выход.", file=sys.stderr)
    sys.exit(1)


def _is_process_alive_windows(pid: int) -> bool:
    """
    Windows-проверка живости PID через kernel32.

    os.kill(pid, 0) на Windows НЕ безопасен: это TerminateProcess — проверка
    живого PID завершила бы проверяемый процесс. Используем OpenProcess
    с минимальным правом SYNCHRONIZE + WaitForSingleObject(handle, 0):

      — OpenProcess вернул NULL:
          GetLastError() == ERROR_INVALID_PARAMETER (87) → процесса нет → False;
          иначе (нет прав)                                  → считаем живым;
      — WaitForSingleObject == WAIT_OBJECT_0 (0)   → процесс завершён → False;
      — WaitForSingleObject == WAIT_TIMEOUT (0x102) → процесс жив → True;
      — прочее (WAIT_FAILED)                        → консервативно True.

    CloseHandle — всегда (finally).
    """
    _SYNCHRONIZE = 0x00100000
    _WAIT_OBJECT_0 = 0x00000000
    _WAIT_TIMEOUT = 0x00000102
    _ERROR_INVALID_PARAMETER = 87

    kernel32 = ctypes.windll.kernel32

    # MINOR-фикс: явные argtypes/restype. Без них дефолтный restype=c_int
    # усекает 64-битный HANDLE на x64 (handle может не влезть в int32).
    # wintypes.HANDLE — c_void_p, корректная разрядность на x86 и x64.
    # Импорт локальный: ctypes.wintypes — отдельный подмодуль, доступный
    # не после любого «import ctypes».
    from ctypes import wintypes as _wintypes
    kernel32.OpenProcess.argtypes = [
        _wintypes.DWORD, _wintypes.BOOL, _wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = _wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [_wintypes.HANDLE, _wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = _wintypes.DWORD
    kernel32.CloseHandle.argtypes = [_wintypes.HANDLE]
    kernel32.CloseHandle.restype = _wintypes.BOOL

    handle = kernel32.OpenProcess(_SYNCHRONIZE, False, int(pid))
    if not handle:
        return kernel32.GetLastError() != _ERROR_INVALID_PARAMETER

    try:
        rc = kernel32.WaitForSingleObject(handle, 0)
        if rc == _WAIT_OBJECT_0:
            return False     # handle сигнален = процесс завершился
        if rc == _WAIT_TIMEOUT:
            return True      # процесс жив
        return True          # WAIT_FAILED / неизвестный код — не убиваем lock
    finally:
        kernel32.CloseHandle(handle)


def _is_process_alive_posix(pid: int) -> bool:
    """
    POSIX-проверка живости PID: os.kill(pid, 0) — сигнал не отправляется.
    ProcessLookupError → процесса нет; PermissionError → процесс есть.
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # процесс существует, но принадлежит другому пользователю
        return True
    except OSError:
        return False


def _is_process_alive(pid: int) -> bool:
    """
    Проверяет, жив ли процесс с данным PID.

    Windows: kernel32.OpenProcess + WaitForSingleObject (НЕ os.kill(pid, 0) —
    на Windows это TerminateProcess, который убил бы проверяемый процесс).
    POSIX: os.kill(pid, 0).
    """
    if sys.platform == "win32":
        try:
            return _is_process_alive_windows(pid)
        except Exception:
            # Не смогли проверить — консервативно считаем живым,
            # чтобы не запустить второй демон поверх первого.
            return True
    return _is_process_alive_posix(pid)


def _release_lock(lock_path: "Path") -> None:
    """
    Освобождает lock: снимает блокировку, закрывает fd, удаляет файл.

    На POSIX unlink делается ДО close (пока мы владелец блокировки) — так
    короче окно, в котором новый процесс успел бы открыть тот же inode.
    На Windows файл нельзя удалить открытым — там unlink идёт после close.
    Повторный unlink после close выполняется ТОЛЬКО на Windows: на POSIX он
    удалил бы уже НОВЫЙ inode, созданный другим процессом в окне между нашим
    unlink и close (два демона одновременно на разных inode).
    Безопасна при отсутствии файла/блокировки.
    """
    global _LOCK_FD
    fd, _LOCK_FD = _LOCK_FD, None

    if fd is None:
        try:
            lock_path.unlink()
        except OSError:
            pass
        return

    try:
        lock_path.unlink()  # POSIX: пока держим flock
    except OSError:
        pass

    _unlock_fd(fd)
    try:
        os.close(fd)
    except OSError:
        pass

    # Windows: удаление возможно только после close. На POSIX второй unlink
    # запрещён — он мог бы удалить inode, созданный другим процессом в окне
    # между unlink и close, и тогда два демона держали бы lock одновременно.
    if sys.platform == "win32":
        try:
            lock_path.unlink()
        except OSError:
            pass


def _signal_handler(signum, frame):
    """
    SIGTERM/SIGINT handler — ТОЛЬКО устанавливает события.

    signal.signal вызывается ТОЛЬКО в main thread (здесь — main).
    Останавливать poller (join) и закрывать pipeline здесь НЕЛЬЗЯ:
    блокирующий join (poll_timeout+5 ≈ 30 с) в хендлере съедал бы весь
    stop_grace_period (45 с) и приводил к SIGKILL. Всё завершение —
    в finally главного цикла.
    """
    print("\n[newsbot] Получен сигнал остановки, завершаем цикл...")
    _shutdown_event.set()
    _digest_requested.set()  # разбудить main thread из wait


def main():
    # Минимальный sanitizer до CLI-ветвления (N4: --check-telegram / --set-chat-id / --check-llm).
    # TokenSanitizer активен даже когда setup_logging ещё не вызван.
    _minimal_sanitizer_handler()

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
        "--bind",
        type=str,
        metavar="OWNER_ID",
        default=None,
        help="Привязать конкретный private-чат по chat_id (только с --set-chat-id)",
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

    # P2-фикс: --bind без --set-chat-id раньше молча игнорировался →
    # запускался демон (пользователь ждал привязку чата). Явная ошибка.
    if args.bind and not args.set_chat_id:
        parser.error("--bind requires --set-chat-id")

    # ── диагностические команды — без полного pipeline ──
    if args.version:
        print(f"newsbot {__version__}")
        sys.exit(0)

    if args.check_telegram:
        _check_telegram()
        return

    if args.set_chat_id:
        if args.bind:
            _set_chat_id_explicit(args.bind)
        else:
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

    pipeline = NewsPipeline(config, shutdown_event=_shutdown_event)

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

    # ── single-instance lock (демон) ───────────────────────────────────────
    # Lock-файл рядом с БД — защита от двойного старта в Docker/на хосте.
    # Pipeline уже создан, db_path известен.
    global _LOCK_FILE
    db_dir = Path(pipeline.storage.db_path).parent
    _LOCK_FILE = db_dir / "newsbot.lock"
    _acquire_lock(_LOCK_FILE, os.getpid())
    logger.info("Single-instance lock acquired: %s", _LOCK_FILE)

    # Minor 3-фикс: ВЕСЬ код после захвата lock — внутри try/finally.
    # Иначе исключение в _maybe_start_poller/signal.signal оставило бы
    # lock-файл на диске (и следующий старт счёл бы демон живым).
    try:
        # ── TelegramPoller: запускается ТОЛЬКО в демон-режиме ──
        # Не запускается в --once / --digest / --check-* (диагностика getUpdates несовместима).
        _maybe_start_poller(config, pipeline, logger)

        # MAJOR-фикс: poll_interval_minutes приводим к int() — config.py уже
        # валидирует/clamp'ит значение («15» → 15, null → дефолт), а здесь
        # страхуемся от нечислового значения: раньше `slept >= interval`
        # падал TypeError'ом ВНЕ try → демон умирал → рестарт-луп.
        try:
            interval_minutes = int(config["poll_interval_minutes"])
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "poll_interval_minutes=%r не число — используем 15 мин.",
                config.get("poll_interval_minutes"),
            )
            interval_minutes = 15
        interval = interval_minutes * 60
        logger.info(
            "Режим: бесконечный цикл, интервал %d мин.",
            interval_minutes,
        )

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        while not _shutdown_event.is_set():
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
                # ждём backoff с проверкой флага (event.wait — прерываемо)
                slept = 0
                while slept < _BACKOFF_SECONDS and not _shutdown_event.is_set():
                    _shutdown_event.wait(1)
                    slept += 1
                continue

            # C3: slept инициализируется КАЖДУЮ итерацию (в т.ч. после успешного цикла)
            slept = 0
            logger.info("Пауза %d мин. до следующего цикла...",
                        interval_minutes)
            # M1: event.wait с таймаутом — реагирует на /digest БЫСТРО
            # (не ждёт 15 мин, просыпается через 30 с или сразу при set)
            while not _shutdown_event.is_set():
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
    finally:
        # ── graceful shutdown: аккуратное завершение ──────────────────────────
        # MAJOR (main)-фикс: единый дедлайн shutdown-фазы. Раньше stop() ждал
        # poll_timeout+5 ≈ 30 с, а затем close() делал ПОВТОРНЫЙ полный join
        # того же потока — суммарно до 60 с при stop_grace_period 45s →
        # SIGKILL до pipeline.close()/_release_lock. Теперь остаток бюджета
        # считается от дедлайна и передаётся в close(join_timeout=...).
        logger.info("Завершение: останавливаем poller...")
        if _poller is not None:
            shutdown_deadline = time.monotonic() + _POLLER_SHUTDOWN_BUDGET_SEC
            remaining = max(0.0, shutdown_deadline - time.monotonic())
            # P1-фикс: stop() получает ОСТАТОК бюджета и делает
            # join(min(poll_timeout+5, remaining)) — раньше join(poll_timeout+5)
            # выполнялся ДО расчёта remaining, и бюджет 40 с не был жёстким.
            _poller.stop(max_join=remaining)
            # close(): остаток дедлайна вместо повторного полного join
            remaining = max(0.0, shutdown_deadline - time.monotonic())
            _poller.close(join_timeout=remaining)
        pipeline.close()
        if _LOCK_FILE is not None:
            _release_lock(_LOCK_FILE)
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
    Читает getUpdates, показывает кандидатов, привязывает private-чат к .env.

    Поведение зависит от TELEGRAM_OWNER_ID:
      — TELEGRAM_OWNER_ID задан: привязывает ТОЛЬКО этот id (и только private).
      — TELEGRAM_OWNER_ID не задан: печатает кандидатов и подсказку --bind <id>.
    Всегда требует chat.type == 'private' (группы/каналы игнорируются).
    """
    import httpx
    import json as _json

    token = telegram_token()
    if not token:
        print("ОШИБКА: TELEGRAM_BOT_TOKEN не найден в .env")
        return

    owner_id = telegram_owner_id()

    if owner_id:
        print(f"TELEGRAM_OWNER_ID задан: {owner_id}")
        print("Режим: привязка ТОЛЬКО к указанному id.")
    else:
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

    # ── Собираем private-чаты (кандидатов) ─────────────────────────
    private_chats = []
    for upd in updates:
        msg = upd.get("message", {})
        chat = msg.get("chat", {})
        if chat.get("type") == "private":
            private_chats.append({
                "update_id": upd.get("update_id", 0),
                "chat_id": chat.get("id"),
                "chat_type": chat.get("type"),
                "username": chat.get("username", ""),
                "first_name": chat.get("first_name", ""),
                "last_name": chat.get("last_name", ""),
            })

    if not private_chats:
        print("Не найдено ни одного private-чата.")
        print("Команда предназначена только для private-чатов (личные сообщения с ботом).")
        return

    # Сортируем по update_id — берём последний (newest)
    private_chats.sort(key=lambda c: c["update_id"], reverse=True)

    # ── TELEGRAM_OWNER_ID задан: привязываем ТОЛЬКО его ────────────
    if owner_id:
        matched = next(
            (c for c in private_chats if str(c["chat_id"]) == str(owner_id)),
            None,
        )
        if matched:
            _write_env("TELEGRAM_CHAT_ID", str(matched["chat_id"]))
            print()
            print(f"✅ TELEGRAM_CHAT_ID={matched['chat_id']} записан в .env")
            print(f"   chat_type: {matched['chat_type']}")
            name = matched["first_name"] or matched["username"] or str(matched["chat_id"])
            print(f"   владелец: {name}")
            print("Запустите --once для проверки отправки.")
        else:
            print()
            print(f"⚠️  В обновлениях нет chat_id={owner_id}")
            print("   TELEGRAM_OWNER_ID задан, но сообщение пришло от другого id.")
            print("   Проверьте, что отправили сообщение СВОЕМУ боту.")
            print()
            print("Кандидаты (private-чаты в обновлениях):")
            for c in private_chats:
                print(f"  id={c['chat_id']}  username={c['username']!r}  "
                      f"name={c['first_name']} {c['last_name']}")
        return

    # ── TELEGRAM_OWNER_ID не задан: печатаем кандидатов ────────────
    print(f"Найдено private-чатов: {len(private_chats)}")
    for c in private_chats:
        name_parts = [p for p in [c["first_name"], c["last_name"]] if p]
        display_name = " ".join(name_parts) or c["username"] or str(c["chat_id"])
        print(f"  update_id={c['update_id']}  id={c['chat_id']}")
        print(f"    type={c['chat_type']}  username={c['username']!r}  name={display_name}")

    print()
    if len(private_chats) == 1:
        c = private_chats[0]
        print(f"Один кандидат: id={c['chat_id']}")
        print("Подтвердить привязку: python -m newsbot --set-chat-id --bind "
              + str(c["chat_id"]))
    else:
        print("Подтвердить привязку: python -m newsbot --set-chat-id --bind <id>")
    print("(Выберите тот, который соответствует ВАШЕМУ аккаунту.)")


def _set_chat_id_explicit(target_id: str) -> None:
    """
    Привязывает конкретный private-чат по его id.
    Вызывается из main() при --set-chat-id --bind <id>.
    """
    import httpx
    import json as _json

    token = telegram_token()
    if not token:
        print("ОШИБКА: TELEGRAM_BOT_TOKEN не найден в .env")
        return

    owner_id = telegram_owner_id()

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
        print(f"Ошибка: {type(e).__name__}")
        return

    if not data.get("ok"):
        print(f"getUpdates вернул ok=false: {data.get('description')}")
        return

    updates = data.get("result", [])
    if not updates:
        print("Нет обновлений. Убедитесь, что отправили сообщение боту.")
        return

    # Ищем private-чат с указанным id
    for upd in updates:
        msg = upd.get("message", {})
        chat = msg.get("chat", {})
        if str(chat.get("id")) == target_id:
            if chat.get("type") != "private":
                print(f"id={target_id} — НЕ private-чат (type={chat.get('type')}).")
                print("Привязка отклонена: только private-чаты.")
                return
            # ── если TELEGRAM_OWNER_ID задан — сверяем ───────────────
            if owner_id and str(chat.get("id")) != str(owner_id):
                print(f"id={target_id} не совпадает с TELEGRAM_OWNER_ID={owner_id}.")
                print("Привязка отклонена.")
                return
            _write_env("TELEGRAM_CHAT_ID", str(chat["id"]))
            print()
            print(f"✅ TELEGRAM_CHAT_ID={chat['id']} записан в .env")
            print(f"   chat_type: {chat.get('type')}")
            name = chat.get("first_name") or chat.get("username") or str(chat["id"])
            print(f"   владелец: {name}")
            print("Запустите --once для проверки отправки.")
            return

    print(f"id={target_id} не найден среди private-чатов в обновлениях.")
    print("Убедитесь, что отправили сообщение боту С ЭТОГО аккаунта.")


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
    pipeline = NewsPipeline(config, shutdown_event=_shutdown_event)
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
        print(f"    [FAIL] Ошибка при проверке баланса: {type(e).__name__}")

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
        # N4: печатаем только тип, не str(e) — может содержать URL с ключом
        print(f"    [FAIL] {type(e).__name__}")


if __name__ == "__main__":
    main()