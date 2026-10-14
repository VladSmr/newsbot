"""
Конвейер: Collector → Dedup → save (state='new') →
  LLM-фильтр → set_state('pending') → delivery →
  set_state('delivered') | отлов ошибки (остаётся 'pending').

State-машина (state, state_at):
  new       — обнаружен, в candidates, ждёт обработки
  pending   — обработка начата (set_state перед send_batch)
  delivered — успешно обработан и доставлен
  skipped   — LLM отфильтровал (score < digest_threshold, retry не трогает)
  deferred  — score 5–6, отложен на дайджест (retry не трогает)
  failed    — терминальный; retry_count исчерпан или recovery на старте цикла
  digested  — терминальный; отправлен в дайджесте

Retry: get_stuck_pending() берёт только 'pending' старше
config.database.retry_threshold_sec (по умолч. 60 с).
Если item уже имеет llm_score — retry НЕ переоценивает.

Двойной send в одном цикле исключён:
  new → pending → delivery. При падении — pending остаётся;
  retry сработает только в СЛЕДУЮЩЕМ цикле.

M2: run_digest может задерживать основной цикл при большом числе
deferred-items (до max_items сообщений с ретраями Telegram API).
Это ожидаемое поведение; на время дайджеста основной цикл блокируется.
"""


import logging
from datetime import datetime, timezone as tz
from typing import List, Optional, Set

from .collector import FeedCollector, FeedResult, NewsItem
from .dedup import filter_new
from .delivery import ConsoleDelivery, TelegramDelivery, Delivery
from .digest import run_digest as _run_digest, build_digest_messages
from .llm import LLMFilter
from .llm.filter import FilterResult, route
from .llm.client import BalanceInfo, LLMError
from .storage import Storage

logger = logging.getLogger("newsbot.pipeline")


class NewsPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.sources = config.get("sources", []) or []

        db_path = self._resolve_db_path(config)
        self.storage = Storage(db_path)

        self.delivery: Delivery = self._build_delivery(config)

        # LLM-фильтр: только если включён И есть api_key
        self.llm_filter: Optional[LLMFilter] = self._build_llm_filter(config)

        # лимит LLM-вызовов на цикл
        self._max_per_cycle = int(config.get("llm", {}).get("max_per_cycle", 150))
        self._llm_calls_left: int = self._max_per_cycle  # общий бюджет на цикл
        # лимит retry-попыток
        self._max_retries = int(
            config.get("database", {}).get("max_retries", Storage.DEFAULT_MAX_RETRIES)
        )

        # кеш фидбэка для few-shot (Фаза 2c); fetched один раз за цикл
        self._feedback_context: Optional[tuple] = None

    def _resolve_db_path(self, config: dict) -> str:
        """
        Резолвит database.path от единого проектного корня (config._find_project_root).
        Гарантирует существование и записываемость директории.
        Не зависит от CWD.
        """
        from . import config as cfg_module

        root = cfg_module._find_project_root()
        rel = config.get("database", {}).get("path", "data/newsbot.db")
        path = root / rel
        # Гарантируем существование и записываемость директории.
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _retry_threshold(self) -> int:
        """Порог retry в секундах из конфига или default."""
        return self.config.get("database", {}).get(
            "retry_threshold_sec",
            Storage.DEFAULT_RETRY_THRESHOLD_SECONDS,
        )

    def _load_feedback_context(self) -> None:
        """
        Загружает последние оценки пользователя из БД для few-shot промпта.
        Вызывается ОДИН раз за цикл (перед началом фильтрации), чтобы
        передать контекст в LLMFilter.evaluate() для каждой новости.

        Гейт: feedback.enabled == True. При отключении — ничего не делает.
        Ошибки получения фидбэка — best-effort (не ломают фильтрацию).
        """
        if not self.config.get("feedback", {}).get("enabled", True):
            self._feedback_context = None
            return

        max_examples = int(
            self.config.get("feedback", {}).get("max_examples", 10)
        )

        dislikes: List[dict] = []
        likes: List[dict] = []

        try:
            dislikes = self.storage.get_recent_feedback(
                verdict="dislike", limit=max_examples
            )
        except Exception as e:
            logger.warning(
                "_load_feedback_context: ошибка получения dislikes: %s", e
            )

        try:
            likes = self.storage.get_recent_feedback(
                verdict="like", limit=max_examples
            )
        except Exception as e:
            logger.warning(
                "_load_feedback_context: ошибка получения likes: %s", e
            )

        # Оба пусты — сбрасываем в None (фильтр не вставляет блок)
        if not dislikes and not likes:
            self._feedback_context = None
        else:
            self._feedback_context = (dislikes, likes)

    def _build_delivery(self, config: dict) -> Delivery:
        token = config.get("telegram", {}).get("token", "") or ""
        chat_id = config.get("telegram", {}).get("chat_id", "") or ""

        if token and chat_id:
            logger.info("Telegram token обнаружен — TelegramDelivery")
            return TelegramDelivery(token=token, chat_id=chat_id)
        else:
            logger.info("Telegram token пуст — ConsoleDelivery")
            return ConsoleDelivery()

    def _build_llm_filter(self, config: dict) -> Optional[LLMFilter]:
        """
        Создаёт LLMFilter если llm.enabled=True и есть api_key.
        Без ключа — логирует warning, возвращает None.
        """
        llm_cfg = config.get("llm") or {}
        if not llm_cfg.get("enabled"):
            logger.info("LLM-фильтр выключен в конфиге")
            return None

        api_key = llm_cfg.get("api_key") or ""
        if not api_key:
            logger.warning(
                "LLM отключён: нет api_key — новости будут слаться непрофильтрованными"
            )
            return None

        try:
            flt = LLMFilter(llm_cfg)
            logger.info("LLM-фильтр инициализирован (%s / %s)",
                         flt.model, flt.push_threshold)
            return flt
        except Exception as e:
            logger.error("Не удалось инициализировать LLM-фильтр: %s", e)
            logger.warning("LLM отключён: новости будут слаться непрофильтрованными")
            return None

    def run_cycle_with_delivery(self) -> None:
        """
        Один цикл сбора + обработки.

        at-least-once семантика:
          1. new-items записываются с state='new' (не pending).
          2. new-items ПОДХВАТЫВАЮТСЯ в начале цикла (recovery).
             Если процесс упал после сохранения new, но до перевода в pending —
             они будут обработаны в следующем цикле (at-least-once ✅).
          3. state переводится в pending перед set_state + доставкой.
          4. delivered выставляется после успешного send_batch.
             ⚠️ ОКНО КРЭША: между return из send_batch и set_state('delivered').
             При падении в этом окне item остаётся 'pending', retry доставит его
             повторно (возможен дубль на стороне TG). Это допустимый риск
             at-least-once; дедуп на стороне TG не реализуется.
        """
        logger.info("=== Начало цикла ===")

        collector = FeedCollector(self.sources)

        # ── K1: подхватываем 'new'-items из прошлого цикла (recovery) ──
        recovered: List[NewsItem] = []
        try:
            recovered = self.storage.get_all_by_state("new")
            if recovered:
                logger.info("K1-recovery: подхвачено %d items со state='new'", len(recovered))
        except Exception as e:
            logger.warning("K1-recovery: ошибка чтения 'new'-items: %s", e)

        total_collected = 0
        total_new = 0
        total_dups = 0
        failed_feeds: List[str] = []
        attempted_feeds = 0
        candidates: List[NewsItem] = []

        # ── Сбор фидов ──
        try:
            for source_name, result in collector.collect():
                if not result.ok:
                    failed_feeds.append(source_name)
                    continue

                attempted_feeds += 1
                items = result.items
                total_collected += len(items)

                new_items, _, dup_count = filter_new(items, self.storage)
                total_dups += dup_count

                # save_many — батчевая вставка (В5); N1 возвращает реально вставленные NewsItem
                new_per_source = 0
                if new_items:
                    inserted_items = self.storage.save_many(new_items)
                    new_per_source = len(inserted_items)
                    total_new += new_per_source
                    candidates.extend(inserted_items)

                logger.info(
                    "[%s] собрано=%d новых=%d дублей=%d",
                    source_name, len(items), new_per_source, dup_count,
                )
        finally:
            # K3: collector.close() ВСЕГДА — даже при исключении.
            collector.close()

        logger.info(
            "=== Итого: собрано=%d новых=%d дублей=%d | фидов: OK=%d failed=%d ===",
            total_collected, total_new, total_dups,
            attempted_feeds, len(failed_feeds),
        )
        if failed_feeds:
            logger.warning("Недоступные фиды: %s", ", ".join(failed_feeds))

        # ── подготовка единого списка items на обработку ──
        # recovered (K1) + candidates (сбор)
        all_items: List[NewsItem] = recovered + candidates
        # обнуляем бюджет на начало доставки
        self._llm_calls_left = self._max_per_cycle

        # ── Few-shot feedback context (Фаза 2c): один раз за цикл ──
        self._load_feedback_context()

        # ── C2: recovery — терминализируем зависшие 'pending' с retry_count>=max ──
        # M3: retry_threshold_sec из конфига (не хардкод), чтобы совпадал с get_stuck_pending
        threshold = self._retry_threshold()
        self.storage.recover_stuck_pending(self._max_retries, retry_threshold_sec=threshold)

        # ── B13: retry-очередь обрабатывается ПЕРЕД кандидатами ──
        # резервируем не менее половины бюджета под retry при наличии таковых
        stuck = self.storage.get_stuck_pending(
            max_age_seconds=threshold,
            max_retries=self._max_retries,
        )

        retry_budget: int
        if stuck:
            retry_budget = max(self._llm_calls_left // 2, len(stuck))
            retry_budget = min(retry_budget, self._llm_calls_left)
            logger.info(
                "B13: retry queue %d items — резервируем бюджет %d/%d; "
                "retry-очередь ПЕРЕД кандидатами (%d new + %d recovered)",
                len(stuck), retry_budget, self._max_per_cycle, len(candidates), len(recovered),
            )
            # C1: _process_batch возвращает НЕВЫПОЛНЕННЫЕ URL
            failed_retry_urls = self._process_batch(stuck, budget_override=retry_budget)
            # инкрементируем retry_count ТОЛЬКО для невыполненных
            if failed_retry_urls:
                self.storage.increment_retry_count(failed_retry_urls)
                # B3: терминализируем только невыполненные, превысившие лимит
                self._check_and_mark_failed(failed_retry_urls)
        else:
            retry_budget = 0

        # ── доставка новых candidates + recovered ──
        remaining_budget = self._llm_calls_left
        if all_items:
            logger.info(
                "Доставка %d candidates+recovered, бюджет %d/%d",
                len(all_items), remaining_budget, self._max_per_cycle,
            )
            # N1: candidates теперь реально вставленные items
            self._process_batch(all_items, budget_override=remaining_budget)

        # ── проверка баланса LLM (раз в check_interval_hours) ──
        self._maybe_check_balance()

        # ── проверка расписания дайджеста (демон-режим) ──
        self._maybe_run_digest()

        # ── очистка ──
        self.storage.purge_seen_titles()
        retention = self.config.get("database", {}).get("retention_days", 90)
        # F1: feedback_targets прунятся отдельно (feedback — обучающие данные, не чистим)
        feedback_ret = self.config.get("feedback", {}).get("retention_days", 180)
        self.storage.purge_items(retention_days=retention, feedback_retention_days=feedback_ret)

    def _process_batch(
        self,
        items: List[NewsItem],
        budget_override: Optional[int] = None,
    ) -> List[str]:
        """
        new/pending → pending → LLM-фильтр (если активен) → route →
        skipped / deferred / delivered.

        Returns:
            Список URL, которые ОСТАЛИСЬ невыполненными (не доставлены,
            не отфильтрованы в skipped/deferred) — только для них имеет
            смысл инкрементировать retry_count и проверять лимит.

        Args:
            items: список NewsItem для обработки
            budget_override: если указано — использовать этот бюджет вместо
                             остатка общего счётчика (для retry с резервацией).

        При ошибке доставки: items остаются 'pending', retry в следующем цикле.
        Если фильтр неактивен (нет api_key) или LLM вернул ok=False →
        всё доставляется непрофильтрованным.

        B3: retry-лимит реализован через retry_count в БД + set_failed();
            get_stuck_pending фильтрует по retry_count < лимита.
        """
        if not items:
            logger.info("Нет items для обработки.")
            return []

        urls = [i.url for i in items]
        self.storage.set_state(urls, "pending")

        # ── LLM-оценка (если фильтр активен и score ещё не сохранён) ──
        skipped_urls: List[str] = []
        deferred_urls: List[str] = []
        to_deliver: List[NewsItem] = []
        llm_batch_records: List[dict] = []

        if self.llm_filter is not None:
            push_th = self.llm_filter.push_threshold
            digest_th = self.llm_filter.digest_threshold

            # лимит LLM-вызовов: бюджет передан явно (retry) или остаток общего счётчика
            budget = budget_override if budget_override is not None else self._llm_calls_left
            items_to_llm = items
            if budget <= 0:
                # бюджет исчерпан — все items остаются pending.
                # C1': НЕ возвращаем как failed_urls → retry_count не инкрементируется,
                # state остаётся pending → retry в следующем цикле.
                logger.warning(
                    "LLM-бюджет цикла исчерпан (0/%d); %d items — pending (retry в след.цикле)",
                    self._max_per_cycle, len(items),
                )
                to_deliver = []  # никто не доставлен, никого не помечаем failed

            if len(items) > budget:
                logger.warning(
                    "LLM-бюджет %d/%d: обрабатываем первые %d items, "
                    "остальные %d — pending (retry в след.цикле)",
                    budget, self._max_per_cycle, budget, len(items) - budget,
                )
                items_to_llm = items[:budget]

            llm_call_count = 0
            for item in items_to_llm:
                # retry: если llm_score уже есть — не переоцениваем (не тратит бюджет)
                if item.llm_score is not None:
                    logger.debug("retry: llm_score=%d уже сохранён для %s",
                                 item.llm_score, item.url[:60])
                    res = FilterResult(
                        ok=True,
                        score=item.llm_score,
                        summary=item.llm_summary or item.summary,
                        reason=item.llm_reason or "",
                    )
                else:
                    # 2.1: try/except Exception — не роняем батч
                    try:
                        res = self.llm_filter.evaluate(
                            item, feedback_context=self._feedback_context,
                        )
                    except Exception as e:
                        logger.warning(
                            "Неожиданное исключение при оценке '%s': %s — "
                            "fallback: шлём непрофильтрованным",
                            item.title[:60], e,
                        )
                        res = FilterResult(ok=False)
                    # сохраняем результат в item (всегда, даже ok=False)
                    item.llm_score = res.score
                    item.llm_summary = res.summary
                    item.llm_reason = res.reason
                    self._llm_calls_left -= 1
                    llm_call_count += 1

                if not res.ok:
                    # F3: fallback → kind='fallback' (не профильтрован LLM,
                    # не путается с обычным push в few-shot выборке (2c)).
                    to_deliver.append(item)
                    item._feedback_kind = "fallback"  # type: ignore[attr-defined]
                    continue

                r = route(res.score, push_th, digest_th)
                if r == "skip":
                    skipped_urls.append(item.url)
                    # 2.2: сохраняем llm-результат и для skipped (единообразие данных)
                    llm_batch_records.append({
                        "url": item.url,
                        "score": res.score,
                        "summary": res.summary,
                        "reason": res.reason,
                    })
                elif r == "digest":
                    deferred_urls.append(item.url)
                    llm_batch_records.append({
                        "url": item.url,
                        "score": res.score,
                        "summary": res.summary,
                        "reason": res.reason,
                    })
                else:
                    # r == "push"
                    to_deliver.append(item)
                    item._feedback_kind = "push"  # type: ignore[attr-defined]
                    llm_batch_records.append({
                        "url": item.url,
                        "score": res.score,
                        "summary": res.summary,
                        "reason": res.reason,
                    })

            logger.debug("LLM-вызовов за батч: %d", llm_call_count)

            # сохраняем все llm-результаты батчем (push + deferred + skipped)
            if llm_batch_records:
                self.storage.update_llm_results_batch(llm_batch_records)

            if skipped_urls:
                self.storage.set_state(skipped_urls, "skipped")
                logger.info(
                    "LLM пропустил %d items (skipped, score<%d)",
                    len(skipped_urls), digest_th,
                )

            if deferred_urls:
                self.storage.set_state(deferred_urls, "deferred")
                logger.info(
                    "LLM отложил %d items (deferred, score %d–%d)",
                    len(deferred_urls), digest_th, push_th - 1,
                )
            # M1: если все отфильтрованы (to_deliver пуст) — НЕ возвращаем
            #     filtered items как failed_urls; сначала фиксируем их state.
            #     Защита: items остаются pending навсегда без этого.
            if not to_deliver:
                return []
        else:
            # фильтр неактивен — всё доставляем непрофильтрованным
            # F3: все — kind='fallback' (не путаются с push в few-shot)
            logger.info("LLM-фильтр неактивен: доставляем все %d items без фильтрации", len(items))
            to_deliver = list(items)
            for item in to_deliver:
                item._feedback_kind = "fallback"  # type: ignore[attr-defined]

        # если все отфильтрованы — нечего доставлять, возвращаем пустой remainder
        if not to_deliver:
            return []

        # ── доставка ──
        # F2: создаём feedback_targets ТОЛЬКО при TelegramDelivery (не Console).
        # F1: get_or_create — идемпотентно (retry не создаёт дублей).
        # F3: kind из item._feedback_kind ('push' или 'fallback').
        # best-effort: ошибка создания target НЕ ломает доставку.
        from .delivery import TelegramDelivery
        should_feedback = bool(
            self.config.get("feedback", {}).get("enabled", True)
            and isinstance(self.delivery, TelegramDelivery)
        )
        if should_feedback:
            for item in to_deliver:
                kind = getattr(item, "_feedback_kind", "push")
                try:
                    item.feedback_target_id = (
                        self.storage.get_or_create_feedback_target_from_item(item, kind)
                    )
                except Exception as e:
                    logger.warning(
                        "feedback: не удалось создать target для '%s': %s",
                        item.title[:60], e,
                    )

        try:
            delivered_urls = self.delivery.send_batch(to_deliver)
            if delivered_urls:
                self.storage.set_state(delivered_urls, "delivered")
                not_delivered = [i.url for i in to_deliver
                                 if i.url not in delivered_urls]
                if not_delivered:
                    logger.info(
                        "Доставлено %d/%d items; %d не доставлено (pending → retry)",
                        len(delivered_urls), len(to_deliver), len(not_delivered),
                    )
            else:
                not_delivered = [i.url for i in to_deliver]
                logger.info("Доставлено 0 items (items остаются pending)")
        except Exception as e:
            logger.error(
                "Доставка упала (%d items → pending, retry в след.цикле): %s",
                len(to_deliver), e,
            )
            not_delivered = [i.url for i in to_deliver]

        # ── формируем множество НЕУСПЕШНЫХ URL ──
        # C1': инкремент/терминализация — ТОЛЬКО для реально не доставленных.
        # Items, срезанные бюджетом, НЕ добавляются: они остаются pending,
        # retry в следующем цикле. При llm_filter is None — всех разобрал
        # to_deliver → processed_urls = all items → excess пуст.
        processed_urls: Set[str] = (
            set(i.url for i in to_deliver) | set(skipped_urls) | set(deferred_urls)
        )
        excess_urls = [
            i.url for i in items
            if i.url not in processed_urls
        ]
        if excess_urls:
            logger.debug(
                "Бюджетный срез: %d items не получили обработки → pending (retry в след.цикле)",
                len(excess_urls),
            )
        # NOTE: excess_urls НЕ добавляются в failed_urls.

        # (a) те, что не доставлены (сбой TG, частичный батч, исключение)
        failed_urls = set(not_delivered)

        return list(failed_urls)

    def _check_and_mark_failed(self, failed_retry_urls: List[str]) -> None:
        """
        C1: переводит реально невыполненные items с retry_count >= лимита
        в терминальный state='failed'. 'failed' не попадает в retry,
        удаляется purge по retention.

        M2: использует mark_failed_if_retries_exhausted (внутри — батчинг,
        фильтрация retry_count >= max_retries, WHERE url IN + AND retry_count >= ?).
        """
        if not failed_retry_urls:
            return
        try:
            marked = self.storage.mark_failed_if_retries_exhausted(
                failed_retry_urls, self._max_retries
            )
            if marked:
                logger.warning(
                    "C1: %d items превысили лимит retry (%d) → state='failed'",
                    marked, self._max_retries,
                )
        except Exception as e:
            logger.warning("_check_and_mark_failed: %s", e)

    def run_digest(self, force: bool = False) -> int:
        """
        Выполняет дайджест для накопленных deferred-items.
        Используется из CLI (--digest) и из демон-цикла.

        M3 (on_empty): force=True всегда выполняет digest.run_digest
        (может отправить пустой дайджест). force=False + пустой список
        → пропускает (on_empty=skip в конфиге).

        Feedback (Фаза 2b-2b): для каждого отправляемого item создаётся
        feedback_target (kind='digest') через get_or_create_feedback_target_from_item
        (идемпотентно). Кнопки 👍/👎 добавляются в сообщение дайджеста.
        Создание target — best-effort, ошибка не ломает дайджест.

        Args:
            force: если True — работает даже при digest.enabled=false
                   и при пустом списке deferred (для ручного --digest).

        Returns:
            Количество отправленных новостей (0 если пропущен/ошибка).
        """
        digest_cfg = self.config.get("digest", {}) or {}
        enabled = digest_cfg.get("enabled", True)

        # M3: --digest (force) работает независимо от enabled
        if not enabled and not force:
            logger.debug("Дайджест выключен (digest.enabled=false)")
            return 0

        max_items = int(digest_cfg.get("max_items", 100))
        max_items_per_message = int(digest_cfg.get("max_items_per_message", 20))
        on_empty = digest_cfg.get("on_empty", "skip")

        # Проверяем, есть ли что отправлять
        deferred = self.storage.get_deferred_items(limit=1)
        if not deferred and not force:
            if on_empty == "skip":
                logger.debug("Дайджест: нет отложенных новостей, пропускаем.")
                return 0

        # ── Feedback targets + передача готовых groups в digest ──
        # F2: только при TelegramDelivery + feedback.enabled
        # m2-фикс: единый источник — groups строятся ОДИН раз в pipeline,
        # передаются в digest._run_digest_with_groups (без пересчёта).
        # Порядок кнопок СТРОГО соответствует порядку новостей в groups.
        should_feedback = bool(
            self.config.get("feedback", {}).get("enabled", True)
            and isinstance(self.delivery, TelegramDelivery)
        )
        deferred_items = self.storage.get_deferred_items(limit=max_items)

        groups, _ = build_digest_messages(
            deferred_items,
            max_items_per_message=max_items_per_message,
        )

        # R1-фикс: НЕ выходи раньше если on_empty="send" и deferred пуст.
        # on_empty="send" должен отправить уведомление "нет новых новостей"
        # даже когда groups пустые (и deferrred пустой).
        if not groups and not (not deferred and on_empty == "send"):
            logger.info("Дайджест: нет сообщений для отправки.")
            return 0

        item_buttons_per_msg: List[List[dict]]
        if should_feedback:
            item_buttons_per_msg = []
            for msg, group_items in groups:
                item_buttons: List[dict] = []
                for item in group_items:
                    try:
                        target_id = self.storage.get_or_create_feedback_target_from_item(
                            item, "digest"
                        )
                    except Exception as e:
                        logger.warning(
                            "feedback: не удалось создать target для '%s': %s",
                            item.title[:60], e,
                        )
                        target_id = None
                    # m1 (T2): URL-whitelist для кнопки — http/https проверяется в telegram.py,
                    # здесь передаём как есть (None если пустой)
                    item_buttons.append({
                        "url": item.url or None,
                        "target_id": target_id,
                    })
                item_buttons_per_msg.append(item_buttons)
        else:
            # ConsoleDelivery или feedback выключен — пустые кнопки
            item_buttons_per_msg = None  # type: ignore[assignment]

        # ── Отправляем через digest (с готовыми groups + buttons) ──
        sent_count, sent_urls = _run_digest(
            storage=self.storage,
            delivery=self.delivery,
            groups=groups,
            item_buttons_per_msg=item_buttons_per_msg,
        )

        # N1: last_digest_at пишется ТОЛЬКО при sent_count>0.
        # При транзиентном сбое (все отправки failed) окно НЕ помечается
        # обработанным — retry в следующем цикле, а не через 12ч.
        if sent_count > 0:
            logger.info("Дайджест: %d новостей → 'digested'", sent_count)
            self.storage.set_meta(
                "last_digest_at",
                datetime.now(tz.utc).isoformat(),
            )
        elif not deferred and on_empty == "send":
            # on_empty=send: отправляем пустое уведомление
            empty_msg = "📰 Дайджест — нет новых новостей."
            try:
                self.delivery.send_text(empty_msg)
                logger.info("Дайджест: пустой дайджест отправлен (on_empty=send)")
                # force+on_empty=send: пустой дайджест отправлен → окно обработано
                self.storage.set_meta(
                    "last_digest_at",
                    datetime.now(tz.utc).isoformat(),
                )
            except Exception as e:
                logger.warning("Дайджест: пустой дайджест не отправлен: %s", e)
        else:
            logger.debug("Дайджест: пропущен (empty, force=%s)", force)

        return sent_count

    def _maybe_run_digest(self) -> None:
        """
        Проверяет расписание дайджеста (digest.times в локальной tz).

        M1: логика «наступило/просрочено». Для каждого времени t из times:
          1. scheduled = ближайший запланированный момент t сегодня
             (или вчера, если now < t и вчера ещё не обработан).
          2. Окно считается «необработанным», если:
               — last_digest_at < scheduled И
               — last_digest_at < (scheduled + 30 мин)
             т.е. дайджест ещё НЕ отправлялся для этого scheduled-окна.
          3. Если now >= scheduled И окно не обработано → запускаем.

        Это гарантирует: нет пропусков (цикл в любую минуту после scheduled
        отработает) и нет дублей (last_digest_at блокирует повтор).
        """
        digest_cfg = self.config.get("digest", {}) or {}
        if not digest_cfg.get("enabled", True):
            return

        times = digest_cfg.get("times", ["09:00", "21:00"])
        if not times:
            return

        now_local = datetime.now(tz.utc).astimezone()
        last_raw = self.storage.get_meta("last_digest_at")
        last_digest: datetime | None = None
        if last_raw:
            try:
                last_digest = datetime.fromisoformat(last_raw).astimezone()
            except Exception:
                pass

        for t in times:
            try:
                sched_hour, sched_min = int(t[:2]), int(t[3:])
            except (ValueError, IndexError):
                logger.warning("Дайджест: невалидное время %r в times", t)
                continue

            # Сегодняшний scheduled момент (local tz)
            sched_today = now_local.replace(
                hour=sched_hour, minute=sched_min, second=0, microsecond=0
            )
            # Вчерашний scheduled момент: для случая когда now ещё не наступил
            # сегодня, но вчерашнее окно ещё не обработано.
            from datetime import timedelta
            sched_yesterday = sched_today - timedelta(days=1)

            # Выбираем ближайший scheduled момент, который <= now_local
            candidates = [c for c in [sched_today, sched_yesterday] if c <= now_local]
            if not candidates:
                # Будущее время — пропускаем
                continue

            # Берём самый поздний из прошедших (actual_scheduled)
            actual_scheduled = max(candidates)
            sched_key = actual_scheduled.strftime("%H:%M")

            # Окно обработано если last_digest_at >= actual_scheduled.
            # Сравниваем через timestamp() — корректно работает с разными tzinfo.
            if (last_digest is not None
                    and last_digest.timestamp() >= actual_scheduled.timestamp()):
                logger.debug(
                    "Дайджест: scheduled=%s уже обработан (last=%s)",
                    sched_key, last_raw,
                )
                continue

            # Окно не обработано: запускаем.
            # N2: запуск возможен даже если цикл попал в окно с задержкой
            # (now > scheduled). last_digest_at блокирует повторный запуск.
            logger.info(
                "Дайджест: scheduled=%s наступил (now=%s), выполняем...",
                sched_key, now_local.strftime("%H:%M"),
            )
            try:
                self.run_digest(force=False)
            except Exception as e:
                logger.error("Дайджест: исключение: %s", e, exc_info=True)
            break  # Одно окно за цикл — достаточно

    def _maybe_check_balance(self) -> None:
        """
        Проверяет баланс LLM-аккаунта не чаще check_interval_hours.
        При низком балансе отправляет алерт владельцу.
        """
        balance_cfg = self.config.get("llm", {}).get("balance") or {}
        if not balance_cfg.get("enabled", True):
            return

        if self.llm_filter is None:
            return

        check_interval_hours = float(balance_cfg.get("check_interval_hours", 24))
        threshold = float(balance_cfg.get("threshold", 2.0))

        # гейтинг по мета-ключу llm_balance_last_check
        last_check_raw = self.storage.get_meta("llm_balance_last_check")
        if last_check_raw is not None:
            try:
                last_check = datetime.fromisoformat(last_check_raw)
                elapsed_hours = (
                    datetime.now(tz.utc) - last_check
                ).total_seconds() / 3600
                if elapsed_hours < check_interval_hours:
                    return
            except Exception:
                pass  # невалидное значение — считаем что не проверяли

        try:
            balance: BalanceInfo = self.llm_filter.get_balance()
        except Exception as e:
            logger.warning("Ошибка get_balance(): %s", e)
            return

        # сохраняем время проверки
        self.storage.set_meta(
            "llm_balance_last_check",
            datetime.now(tz.utc).isoformat(),
        )

        if balance.is_available and balance.total_balance < threshold:
            # cooldown: не чаще check_interval_hours
            last_alert_raw = self.storage.get_meta("llm_balance_last_alert")
            should_alert = True
            if last_alert_raw is not None:
                try:
                    last_alert = datetime.fromisoformat(last_alert_raw)
                    elapsed_hours = (
                        datetime.now(tz.utc) - last_alert
                    ).total_seconds() / 3600
                    if elapsed_hours < check_interval_hours:
                        should_alert = False
                except Exception:
                    pass  # невалидное значение — шлём алерт

            if should_alert:
                currency = balance.currency or "USD"
                alert_text = (
                    f"⚠️ Низкий баланс DeepSeek: {balance.total_balance} {currency} "
                    f"(порог {threshold}). Пополни: https://platform.deepseek.com"
                )
                if self.delivery.send_text(alert_text):
                    self.storage.set_meta(
                        "llm_balance_last_alert",
                        datetime.now(tz.utc).isoformat(),
                    )
                    logger.info("Алерт о низком балансе DeepSeek отправлен")

    def close(self) -> None:
        self.storage.close()
        self.delivery.close()
        # закрываем LLM-клиент (идемпотентно)
        if self.llm_filter is not None:
            try:
                self.llm_filter.client.close()
            except Exception:
                pass