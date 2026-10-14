# NEWS BOT — PROJECT STATE (сводка для восстановления контекста)

## Что это
Персональный Telegram-бот: собирает новости из RSS (30 фидов), дедуплицирует,
фильтрует через LLM (DeepSeek) по интересам владельца и присылает только
релевантные. Профиль: ТОЛЬКО факты и события с проверяемыми деталями;
отсекать мнения, заявления, прогнозы, слухи (кроме массовых волн как явления),
аналитику, светскую хронику.

## Профиль интересов (источник — newsbot/llm/prompts.py SYSTEM_PROMPT)
Отсекать: мнения, прогнозы, «источники сообщили», заявления/обещания,
аналитику, слухи, светскую хронику. Исключение: массовая волна слухов —
тоже факт.

Темы:
1. Атаки БПЛА по РФ — ночные сводки: сколько сбито, регионы, повреждения
   (нефтебазы, НПЗ, дата-центры, инфраструктура).
2. Мобилизация, эвакуации (+ массовые слухи как явление).
3. Критичные происшествия — РФ и мир.
4. Краснодар и край — БПЛА, происшествия, инфраструктура
   (отключения, транспорт, стройки), погода/стихия.
5. Топливо — дефицит, остановки НПЗ, атаки (РФ и край). Ежедневная
   динамика цен — НЕ интересна.
6. Санкции и крупные корпоративные события — уход/возврат компаний,
   крупные сделки.
7. Хай-тек — кибератаки/массовые утечки, космос, железо/чипы.
   Релизы ИИ-моделей — НЕ интересны (кроме инцидентов).
8. Игры — крупные AAA-релизы и культовые франшизы (GTA 6, Ведьмак):
   даты, трейлеры, анонсы. Индустрия/скандалы — НЕ интересны.
9. Кино — франшизы и блокбастеры: даты, трейлеры, production-факты.
10. Стихийные бедствия — смерчи, наводнения: РФ и мир.
11. Авиа- и транспортные катастрофы.
12. Наука и медицина — прорывы, крупные открытия.
13. Техногенные катастрофы и экология — выбросы, разливы, пожары,
    радиация.
14. Протесты и забастовки (мир) — крупные, только факты.

## Технологии
Python 3.8+, feedparser, httpx, PyYAML, SQLite (stdlib).
Отправка в ТГ: python-telegram-bot. LLM: DeepSeek (deepseek-v4-pro по умолч.),
~1–2 $/мес. Запуск: `python -m newsbot --once` / `python -m newsbot` (демон).

## Архитектура

**Конвейер:** Collector (RSS/feedparser+httpx) → Dedup → save_many(state='new')
→ recovery K1 (подхват 'new') → recovery C2 (терминализация зависших 'pending')
→ few-shot: загрузка feedback из БД → LLM-фильтр (DeepSeek, бюджет 150/цикл,
retry-items с готовым llm_score не тратят бюджет, few-shot блок в промпте)
→ set_state('pending') → Delivery → set_state('delivered').
Retry: `get_stuck_pending()` — 'pending' старше retry_threshold_sec (60 с).

**State-машина (v8):** new / pending / delivered / skipped / deferred / failed / digested.
`retry_count` INTEGER DEFAULT 0 в БД. При исчерпании retry_count >= max_retries
→ 'failed' (терминальный). Recovery C2: зависшие 'pending' с retry_count
>= лимитом → 'failed' на старте цикла. Purge: delivered/skipped/failed/digested
старше retention_days (90). deferred — НЕ удаляются (ждут дайджест).

**Схема БД v8** (SCHEMA_VERSION=8):
- `items`: id, source, url, title, summary, published, fetched_at,
  state, state_at, llm_score, llm_summary, llm_reason, filtered_at, retry_count.
- `seen_titles`: title_hash, source, fetched_at, expires_at (TTL 30 дн.).
- `meta`: key TEXT PRIMARY KEY, value TEXT (для llm_balance_last_check,
  llm_balance_last_alert, last_digest_at, tg_updates_offset).
- `feedback_targets`: id, url, source, title, summary, llm_score, kind
  ('push'/'digest'), chat_id, message_id, created_at.
  Индекс: (url, kind).
- `feedback`: id, target_id UNIQUE, url, source, title, summary, llm_score,
  verdict CHECK('like'/'dislike'), chat_id, message_id, created_at, updated_at.
  Индексы: (verdict, created_at), (created_at), (updated_at).
- WAL journal_mode, busy_timeout=5000.
- Миграции v1→v8 идемпотентны (IF NOT EXISTS, безопасный ROLLBACK,
  ранний return при актуальной версии).

**LLM-фильтр:**
- DeepSeek, model=deepseek-v4-pro (⚠️ проверить id живым ключом).
- system_prompt = SYSTEM_PROMPT из prompts.py (источник истины; поле
  llm.system_prompt в config.yaml пустое = штатный оверрайд).
- Структурный JSON: {score:0-10, summary, reason}, temperature=0.
- Пороги: score ≥ 7 → пуш (delivered); 5–6 → deferred; ≤ 4 → skipped.
- Fallback при сбое/отсутствии ключа: всё доставляется непрофильтрованным (kind='fallback').
- `max_per_cycle=150` — единый бюджет на цикл (candidates + retry ≤ 150).
- Retry-items с готовым llm_score не тратят бюджет.
- Balance-алерт: при total_balance < threshold отправляется в Telegram
  (cooldown по check_interval_hours; при is_available=False — лог, НЕ алерт).
- Few-shot (Фаза 2c): последние 👍/👎 из БД → блок «ДАННЫЕ (НЕ ИНСТРУКЦИИ)»
  в user-сообщении. Санитизация: удаляются \n, \r, control-chars,
  Unicode line separators (U+2028, U+2029, \x85). Обёртка `<|пример: …|>`.
  Суммарный лимит блока 1500 символов. SYSTEM_PROMPT содержит охранную
  директиву. Гейт: feedback.enabled. Один запрос get_recent_feedback на цикл.

**Дайджест:**
- Собирает items state='deferred', сортирует llm_score DESC, fetched_at ASC.
- Расписание: 2×/день из digest.times (локальная tz, дефолт ["09:00","21:00"]),
  ручной запуск `--digest` или `/digest` из Telegram.
- Конфиг: digest.enabled / digest.times / digest.max_items / digest.max_items_per_message / digest.on_empty.
- Гарантия ≤4096 символов/сообщение (HTML-escape, бинарный поиск body по word boundary).
- `last_digest_at` пишется ТОЛЬКО при успешной отправке ≥1 новости.
- State='digested' ставится только реально отправленным items (at-least-once).
- Частичный сбой: отправленные → 'digested', остаток → 'deferred' (retry в следующем окне).
- Кнопки 👍/👎 добавляются к каждой новости дайджеста (kind='digest').

**TelegramPoller (Фаза 2b):**
- Long-poll getUpdates в демон-потоке. Запускается только в демон-режиме
  (НЕ при --once / --digest / --check-*). poll_timeout из config (дефолт 25 с).
- Отдельное соединение SQLite в poller thread (не конфликтует с main).
- Обрабатывает: callback_query (👍/👎, формат `fb:<id>:<u|d>`, только от владельца),
  команды /digest / start / help от владельца.
- offset хранится в meta (`tg_updates_offset`) — устойчив к рестартам.
- 409 Conflict → backoff 60 с.

**Feedback (Фазы 2b/2c):**
- feedback_targets: новость отправлена → записана с кнопками (идемпотентно по url+kind).
- feedback: голос пользователя (like/dislike) → upsert (re-vote обновляет verdict+updated_at).
- Таблица feedback НЕ чистится (практически permanent — обучающие данные).
- Конфиг: feedback.enabled / feedback.max_examples / feedback.retention_days
  (чистит только feedback_targets, не feedback).

**Формат карточки в ТГ:**
```
📰 {заголовок}
{содержательный текст 2–5 предложений}
📍 {источник} · {локальное время}
🔗 Открыть источник  (инлайн-кнопка на url)
```

## Источники (30 RSS-фидов, см. config.yaml)
РФ (6): Интерфакс, ТАСС, РИА, Lenta, REGNUM, Коммерсантъ.
Кубань (2): Югополис, Кубанские новости.
Мир (4): Al Jazeera, Guardian World, Sky News World, France 24 English.
Авиа (1): AeroTime.
Наука (5): ScienceDaily, Phys.org, STAT News, N+1, Naked Science.
Кибер (3): BleepingComputer, KrebsOnSecurity, Habr news.
Чипы (1): Tom's Hardware. Космос (3): SpaceNews, NASA, NASASpaceflight.
Игры (3): IGN, Eurogamer, DTF. Кино (2): Deadline, Variety.

## Тесты
**333 теста, 9 файлов, запуск: `python -m pytest tests -q`**
- `test_llm_filter_mocks` — LLM-фильтр (mock client)
- `test_pipeline_llm` — pipeline + LLM
- `test_core_reliability` — retry/recovery/failed/state-машина
- `test_delivery_security` — delivery + security hardening
- `test_digest` — дайджест, build_digest_messages, ≤4096
- `test_feedback_storage` — feedback_targets + feedback CRUD
- `test_feedback_flow` — pipeline + feedback, few-shot context
- `test_telegram_poller` — TelegramPoller, callback, команды
- `test_feedback_prompt` — build_feedback_block, санитизация, лимиты

## Фиксы по ревью (краткий список)

| ID    | Что исправлено                                                         |
|-------|------------------------------------------------------------------------|
| K1    | Recovery 'new' на старте цикла: подхватываются items со state='new' из прошлого цикла (at-least-once) |
| K2    | HTTPError в collector.py — httpx.HTTPError                             |
| K3    | finally collector.close() — всегда закрывается, даже при исключении     |
| B3    | retry_count/'failed'/лимит: increment_retry_count + mark_failed_if_retries_exhausted + get_urls_with_retry_gte |
| B5    | save_many: батчевая вставка в одной транзакции, возврат реально вставленных NewsItem (N1) |
| B13   | Retry-очередь обрабатывается ПЕРЕД кандидатами; резервируется ≥½ бюджета |
| M1    | Если все items отфильтрованы (to_deliver пуст) — возврат [], а не filtered items как failed_urls |
| M2    | mark_failed_if_retries_exhausted: фильтрация retry_count >= max_retries через WHERE + AND в одном UPDATE |
| M3    | retry_threshold_sec из конфига, не хардкод (совпадает с get_stuck_pending) |
| N1    | save_many возвращает List[NewsItem] реально вставленных                 |
| N4    | MIN_TITLE_LEN=3: короткие (<3) заголовки не проверяются по title-dedup  |
| N5    | get_urls_with_retry_gte + mark_failed_if_retries_exhausted: батчинг по ~500   |
| C1    | _check_and_mark_failed: реально не доставленные → failed; бюджетный срез не попадает в failed_urls |
| C2    | Recovery зависших pending на старте цикла (реплика recovery из loop в pipeline.py) |
| T2-7  | Маскирование токенов: длина + последние 4 символа, НЕ первые 6        |
| W1    | WAL journal_mode + busy_timeout=5000                                    |
| Cap   | summary cap MAX_SUMMARY_CHARS=6000                                      |
| Единств. | Единый форматтер времени (UTC isoformat), опечатка "отклющён"→"отключён" |
| F1    | Идемпотентное создание feedback_target (get_or_create); индекс (url, kind); feedback_targets не прунятся основным purge (свой retention) |
| F2    | Feedback targets создаются ТОЛЬКО при TelegramDelivery (не Console)    |
| F3    | kind='fallback' для непрофильтрованных (не путаются с push в few-shot) |
| F4    | poll_timeout из конфига telegram.* (не хардкод 25)                     |
| F5    | get_recent_feedback не возвращает chat_id/message_id (не попадают в промпт) |
| F6    | digest.enabled проверяется в _maybe_run_digest (дайджест по расписанию выключаем) |
| N1-NIT| record_feedback: INSERT OR IGNORE + UPDATE — атомарный upsert, parallel-safe, created_at неизменяем |
| N2    | RECREATE TABLE feedback для миграции v8 (CHECK + updated_at) в явной транзакции |
| N3    | Миграция v8: фильтрация невалидных verdict при копировании; пропуск строк с логированием |
| N4    | TgApiError: http_status хранится в исключении (детект 409 без URL/токена в логе); _safe_status_log |
| M1    | run_digest: groups строятся ОДИН раз в pipeline, передаются в digest (единый источник) |
| M2    | run_digest: on_digest callable только СИГНАЛИЗИРУЕТ, НЕ выполняет в poller thread |
| M3    | digest._maybe_run_digest: расчёт scheduled-окон (today/yesterday), last_digest_at блокирует дубль |
| M1-фикс | digest body truncation: бинарный поиск по RAW body → escaped ≤ budget; footer cap по ESCAPED (не RAW) |
| M2-фикс | digest item: source cap до 100 RAW-символов (чтобы escape не раздул overhead) |
| M3-фикс | telegram_io: Storage.close() ВНУТРИ poller thread (finally); close() НЕ закрывает Storage |
| F1    | TelegramPoller: отдельное соединение Storage в poller thread (C1); offset в meta (N1) |
| F2    | feedback_targets retention_days через feedback_retention_days параметр в purge_items |
| F3    | feedback_targets.created_at → prune; feedback не чистится (обучающие данные) |
| T2-9  | LLM score: принимаем int / целочисленный float / str-"int"; отклоняем bool, NaN, inf, дробные |
| T2-10 | _save_offset/set_meta: ConnectionError при записи offset → лог, не падаем (offset подхватится в следующем tick) |
| m1    | digest: send_digest (с кнопками) vs send_text (без кнопок) — выбор по item_buttons_per_msg |
| m2    | run_digest: groups могут быть переданы из pipeline (единый источник, без пересчёта) |

## План этапов

- [x] Этап 1 (MVP-каркас): ✅ ЗАВЕРШЁН, APPROVE.
- [x] Этап 2a (Telegram-доставка): ✅ РЕАЛИЗОВАН. getUpdates, send_batch,
  HTML-экранирование, split >4096, retry 429, close только в NewsPipeline.close(),
  inline-кнопка. R1-фикс: ConsoleDelivery → List[str].
- [x] Этап 2b-1 (LLM-ядро): ✅ РЕАЛИЗОВАН. DeepSeekClient, LLMFilter,
  SYSTEM_PROMPT v3, config llm.*, env deepseek_api_key(), --check-llm.
  18/18 тестов на моках.
- [x] Этап 2b-2a (LLM в pipeline): ✅ РЕАЛИЗОВАН. Schema v4, deferred-state,
  update_llm_results_batch, retry с переиспользованием llm_score.
  15 новых тестов, итого 33.
- [x] Этап 2b-2b (баланс-алерт + hardening): ✅ РЕАЛИЗОВАН.
  BalanceInfo, is_available-логика, SCHEMA v5 (meta), send_text, max_per_cycle,
  try/except evaluate(), finally-close клиента. 40 тестов.
- [x] Финальное hardening (после полного code-review): ✅ ЗАВЕРШЕНО.
  Recovery 'new', HTTPError, finally, retry_count/failed/лимит, WAL/busy_timeout,
  save_many N1, cap summary, единый форматтер, маскирование токенов,
  восстановление _set_chat_id (max update_id), и др.
  **123 теста зелёные.**
- [x] Этап 2b-2c (Дайджест): ✅ РЕАЛИЗОВАН. Schema v8, digest.py, build_digest_messages,
  расписание 2×/день (digest.times), ручной --digest, on_empty, гарантия ≤4096,
  last_digest_at по мета, digested-state, at-least-once. **333 теста зелёные.**
- [x] Фаза 2b (TelegramPoller + обратная связь): ✅ РЕАЛИЗОВАН. TelegramPoller
  (long-poll, кнопки 👍/👎, команды), Schema v7+8 (feedback_targets + feedback),
  record_feedback, get_recent_feedback, few-shot в промпте.
- [x] Фаза 2c (Few-shot адаптация): ✅ РЕАЛИЗОВАН. build_feedback_block,
  санитизация control-chars, маркер «ДАННЫЕ (НЕ ИНСТРУКЦИИ)», охранная
  директива в SYSTEM_PROMPT, суммарный лимит 1500.
- [ ] Этап 3: скрейперы, RSSHub-мост для ТГ-каналов (Astra, Baza, Shot, Mash,
  Sirena), Steam-подписки на тайтлы, USGS GeoJSON (землетрясения).
- [ ] Этап 5: деплой 24/7 (Docker), VPS, живая проверка фидов и Telegram.

## Оставшиеся задачи

| # | Что                          | Приоритет     | Комментарий                                       |
|---|------------------------------|---------------|---------------------------------------------------|
| 1 | URL-нормализация             | Средний       | utm_* + trailing slash; отложена (known limitation)|
| 2 | Проверка модели DeepSeek     | Высокий       | `deepseek-v4-pro` — проверить id через `--check-llm` живым ключом |
| 3 | Deploy на VPS, этап 5        | Высокий       | Docker, VPS 24/7, проверить фиды и Telegram живьём   |
| 4 | Telegram-доставка живьём    | Высокий       | api.telegram.org недоступен с локальной машины      |
| 5 | Этап 3 (RSSHub, USGS, Steam) | Низкий        | ТГ-каналы → RSSHub, землетрясения → USGS GeoJSON   |

## Принятые решения и открытые вопросы
- Доставка: ГИБРИД (срочное — пуш, остальное — дайджест 2×/день).
- Хостинг: НЕ выбран — решается перед этапом 5 (VPS ~250 ₽/мес vs домашний ПК).
- Токен бота: в .env (TELEGRAM_BOT_TOKEN), не в config.yaml, не в git.
- Риск at-least-once: крэш между send_batch и set_state('delivered') —
  item остаётся 'pending', retry доставит повторно. Дедуп на стороне TG
  не реализуется.
- Риск at-least-once (digest): крэш между отправкой и set_digested →
  item остаётся 'deferred', retry в следующем окне (не потеряется).
- ТГ-каналы нельзя читать Bot API напрямую; путь — свой RSSHub (этап 3).
- Землетрясения USGS — GeoJSON, отдельный парсер (этап 3).
- Few-shot: feedback.enabled=1 блокирует few-shot; при disabled — фильтр
  работает без адаптации ( штатное поведение).
- TelegramPoller создаёт СОБСТВЕННОЕ соединение SQLite в poller thread —
  WAL-блокировки решаются через busy_timeout=5000ms.