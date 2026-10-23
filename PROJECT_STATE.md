# NEWS BOT — PROJECT STATE (сводка для восстановления контекста)

## Репозиторий

**Публичный:** https://github.com/VladSmr/newsbot
Основная ветка на GitHub: `main` (локально — `master`).

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
Python 3.12, feedparser, httpx, PyYAML, SQLite (stdlib).
Отправка в ТГ: httpx + Telegram Bot API. LLM: DeepSeek (deepseek-flash по умолч., deepseek-v4-pro — альтернатива),
~1–2 $/мес. Запуск: `python -m newsbot --once` / `python -m newsbot` (демон).

## Архитектура

**Конвейер:** Collector (RSS/feedparser+httpx) → Dedup (кросс-источниковый,
двухуровневый) → save_many(state='new') → recovery K1 (подхват 'new')
→ recovery C2 (терминализация зависших 'pending') → few-shot: загрузка
feedback из БД → LLM-фильтр (DeepSeek, бюджет 150/цикл, retry-items с готовым
llm_score не тратят бюджет, few-shot блок в промпте) → set_state('pending')
→ Delivery → set_state('delivered'). Retry: `get_stuck_pending()` — 'pending'
старше retry_threshold_sec (60 с).

**Кросс-источниковая дедупликация (v9, фикс ложных склеек):**
- Уровень 1 (глобальный точный): `title_hash` БЕЗ `source` в пределах
  `window_days` (дефолт 3 дня), активируется `cross_source=true`.
- Уровень 2 (fuzzy): `SequenceMatcher` с предфильтром (токен-сеты/длина) и
  лимитом 200 кандидатов — O(N), не O(N²); активируется `fuzzy=true`,
  **только при `cross_source=true`**; порог `threshold`=0.95 (поднят с 0.9).
  Дубли → `skipped` (URL-дедуп в следующих циклах).
- **Защита от ложных склеек** — формула `дубль ⟺ (ratio≥threshold) И (числа совпадают)
  И (Jaccard≥token_threshold=0.8)**:
  - числовой guard: мультимножества (кратности учитываются) `\d+(?:[.,]\d+)?` должны совпадать (жёсткий запрет
    при расхождении — «Bitcoin упал до 60000» vs «до 30000» не склеятся);
  - token-guard: Jaccard значащих токенов (≥4 буквы, без стоп-слов/чисел) ≥ 0.8
    («apple announces X» vs «google announces X» не склеятся, разные сущности).
- Нормализация: lowercase, collapse spaces, strip, удаление пунктуации,
  срез префиксов `Reuters:`, `Bloomberg:`, `Источник:` и т.п.
- При `cross_source=false` fuzzy игнорируется — сохранено прежнее per-source поведение.
- Хранилище для fuzzy: колонка `title_normalized` в `seen_titles` (идемпотентная
  миграция v8→v9, `ALTER TABLE ADD COLUMN IF NOT EXISTS`).

**LLM-дедупликация (семантическая, «одно событие — разные формулировки»):**
  *Отдельный лёгкий LLM-вызов (только заголовки) ДО скоринга.*
  Запускается через `LLMFilter.detect_duplicates(items, recent_titles)` → `Set[int]`
  (0-based индексы дублей). Ключ `llm.llm_dedup: true/false` (дефолт True).
  Конфиг: `dedup.llm_recent_limit: 20` (recent titles в контексте), `window_days: 3`
  (источник для recent из items table).
  - Критерий: «одно и то же событие, разные слова/язык/числа».
    Примеры: RU-пара «сбили 80 дронов» (Интерфакс) ↔ «сбили 80 БПЛА» (ТАСС);
    EN-пара «34 children killed» (Guardian) ↔ «dozens killed» (Sky News).
  - Промпт: `DEDUP_SYSTEM_PROMPT` + `build_dedup_user_message()` (prompts.py).
    Ответ: `{"duplicates": [1, 3, ...]}` — 1-based индексы основного списка.
    Делимитеры `<|данные:N|>` защищают от injection (sanitize_item_field).
  - Чанкование: по 40 кандидатов на вызов; `recent` подмешивается в каждый чанк.
  - Устойчивость: `LLMError` → пустое множество (безопасно, ни один дубль не пропущен);
    битые/несуществующие индексы → игнорируются; `<2` кандидатов → вызова нет.
  - Интеграция в pipeline: вызывается один раз за цикл (ДО скоринга) в
    `run_cycle_with_delivery()`, после сбора `candidates` и `save_many`.
    Найденные дубли → `storage.save_skipped()` (URL-дедуп) + исключаются из `all_items`.
  - `storage.get_recent_titles(window_days, limit, exclude_source) → List[(title, source)]`:
    читает оригинальные (не нормализованные) заголовки из `items` table
    (исключая `skipped/failed`), лимит 20 по умолчанию.

**State-машина (v13):** new / pending / delivered / skipped / deferred / failed / digested.
`retry_count` INTEGER DEFAULT 0 в БД. При исчерпании retry_count >= max_retries
→ 'failed' (терминальный). Recovery C2: зависшие 'pending' с retry_count
>= лимитом → 'failed' на старте цикла.
Purge: 'new' старше 7 дней + delivered/skipped/failed/digested старше retention_days (1 день).
'deferred': отдельное окно `deferred_retention_days` (3 дня) — старые неотправленные
отложенные удаляются (критично для VPS 1 GB RAM; раньше копились бесконечно).
**Осознанное решение владельца:** retention_days=1 и deferred_retention_days=3
(не 90/14) — старые новости не нужны, хранилище экономится.
'pending' — не удаляется (активный retry).
K1-recovery ограничен `recovery_new_limit` (500) — при долгой паузе 'new' накапливались
и загружались в RAM целиком.
2c-фикс: remainder кандидатов (не доставленные по сбою TG) инкрементируют retry_count
и терминализируются при исчерпании лимита — at-least-once для candidates.
skipped: LLM-фильтр (score<5) + кросс-источниковые дубли (уровни 1/2).

**Схема БД v14** (SCHEMA_VERSION=14):
- `items`: id, source, url, **url_norm TEXT (v14)**, title, summary, published, fetched_at,
  state, state_at, llm_score, llm_summary, llm_reason, **llm_urgent INTEGER 0/1 (v10/v11)**,
  filtered_at, retry_count, **tags TEXT (v4), cluster_key TEXT (v12), lang TEXT (v13)**.
  `llm_urgent` — персистентный флаг срочности; восстанавливается на retry.
  Legacy-строки (до v10) с `llm_urgent=NULL` → `urgent=False` (безопасный дефолт).
  `tags` — json-список `<category>` из фида (в `cluster_key` не входит).
  `cluster_key` = `{location}|{stem_hash}|{pub_day}` (fallback: `nostem:{md5(title)}`);
  используется для 2-часового доменного дедупа (всегда включён, не конфигурируется).
- `seen_titles`: title_hash, source, fetched_at, expires_at,
  title_normalized (TTL 30 дн.; v9 для fuzzy).
- `meta`: key TEXT PRIMARY KEY, value TEXT (для llm_balance_last_check,
  llm_balance_last_alert, last_digest_at, tg_updates_offset, **paused**).

**Пауза (`/stop` / `/continue`):**
- Флаг `paused`="1"/"0" в `meta` — виден обоим потокам (поллер и main cycle).
  Хранится в БД → переживает рестарт контейнера.
- При паузе: RSS-сбор и сохранение → `state='new'` (продолжается);
  LLM-фильтр, доставка, дайджест, алерт баланса — пропускаются.
  Purge (seen_titles, items) выполняется.
- Mid-batch: проверка `_is_paused()` перед каждым item → break,
  необработанные items остаются `pending` (retry в след.цикле, at-least-once).
- Пауза приоритетнее off-peak гейта (проверяется раньше).
- `/stop`/`/continue` обрабатываются в поллер-потоке; `/digest` при паузе
  возвращает заглушку (`_on_digest` не дёргается).
- TelegramPoller стартует при наличии `telegram.token`+`chat_id`
  НЕЗАВИСИМО от `feedback.enabled` (фидбэк-обработка гейтится
  по параметру `feedback_enabled` в конструкторе).
- `busy_timeout=5000` для конкурентной записи meta из двух соединений
  (WAL: читатели не блокируют, писатели не блокируют читателей).
- `feedback_targets`: id, url, source, title, summary, llm_score, kind
  ('push'/'digest'), chat_id, message_id, created_at.
  Индекс: (url, kind).
- `feedback`: id, target_id UNIQUE, url, source, title, summary, llm_score,
  verdict CHECK('like'/'dislike'), chat_id, message_id, created_at, updated_at.
  Индексы: (verdict, created_at), (created_at), (updated_at).
- WAL journal_mode, busy_timeout=5000.
- Миграции v1→v11 идемпотентны (IF NOT EXISTS, безопасный ROLLBACK,
  ранний return при актуальной версии).

**LLM-фильтр:**
- DeepSeek, model=deepseek-flash (⚠️ проверить id живым ключом; deepseek-v4-pro — альтернатива).
- system_prompt = SYSTEM_PROMPT из prompts.py (источник истины; поле
  llm.system_prompt в config.yaml пустое = штатный оверрайд).
- **Батч-оценка** (`llm.batch_size=10`, `llm.max_tokens=4096`):
  - ≤10 новостей одним LLM-вызовом — system prompt платится раз на батч (экономия input-токенов).
  - `BATCH_SYSTEM_PROMPT` + `build_batch_user_message` в prompts.py: items нумеруются индексами 1..N, маркировка `<|данные:ID|>` — injection-safe.
  - Формат ответа: `{"results": [{"id":1,"score":0-10,"urgent":bool,"summary":"...","reason":"..."},...]}`.
  - Per-item fallback: пропущенный/битый id или невалидное поле → только этот item `ok=False` (остальные обрабатываются).
  - Полный сбой (`LLMError`) → все items `ok=False`.
  - При `batch_size=1` — поштучный режим, вызывается `evaluate()`.
  - `evaluate_batch()` в LLMFilter, `_parse_batch_response()` — per-item валидация (tolerant score/urgent).
  - Retry-items (llm_score уже есть) синтезируют FilterResult из БД, НЕ попадают в батч, НЕ тратят бюджет.
- Структурный JSON: {score:0-10, summary, reason, urgent:bool}, temperature=0.
- Флаг `urgent`: True — война/вторжение, мобилизация, катастрофа с массовыми жертвами (десятки+), ядерный инцидент; при сомнении → False (безопасный дефолт).
- Маршрутизация (`push_urgent_only=true`, дефолт):
  `urgent AND score>=7` → пуш (delivered); `score 5–6` → deferred (дайджест); ≤ 4 → skipped.
  `urgent AND score<7` → deferred (НЕ пуш; вторичная защита от перестарания модели).
- Маршрутизация (`push_urgent_only=false`, обратная совместимость): прежнее поведение — score ≥ 7 → пуш.
- Конфиг: `llm.push_urgent_only` (bool, дефолт True), `llm.push_threshold` (7), `llm.digest_threshold` (5), `llm.batch_size` (10), `llm.max_tokens` (4096).
- Fallback при сбое/отсутствии ключа: всё доставляется непрофильтрованным (kind='fallback').
- `max_per_cycle=150` — единый бюджет на цикл (candidates + retry ≤ 150).
- Retry-items с готовым llm_score не тратят бюджет; **urgent восстанавливается из llm_urgent (v10/v11)**.
  Legacy-строки с `llm_urgent=NULL` → `urgent=False`.
- Balance-алерт: при total_balance < threshold отправляется в Telegram
  (cooldown по check_interval_hours; при is_available=False — лог, НЕ алерт).
**Off-peak гейт (llm.off_peak_only):**
- `off_peak_only: false` (по умолчанию) — без ограничений (текущее поведение).
- `off_peak_only: true` — LLM вызывается только в off-peak часы DeepSeek (×0.5).
  Пик UTC (Пн–Пт): 01:00–03:59 и 06:00–09:59. MSK: 04:00–06:59 и 09:00–12:59.
  В пик: RSS собирается, items остаются необработанными (`new`; подхватятся
  в следующем off-peak цикле). Ручной запуск `--once` тоже гейтится (консистентно).
  Китайские гос. праздники консервативно считаются пиком.

**Few-shot (Фаза 2c):** последние 👍/👎 из БД → блок «ДАННЫЕ (НЕ ИНСТРУКЦИИ)»
  в user-сообщении. Санитизация: удаляются \n, \r, control-chars,
  Unicode line separators (U+2028, U+2029, \x85). Обёртка `<|пример: …|>`.
  **M1-фикс:** title/summary/source прогоняются через `sanitize_item_field()` —
  удаление control-chars + нейтрализация делимитеров (``<|`` → ``< |``,
  ``|>`` → ``| >``). Работает в обоих путях: одиночном (`LLMFilter.evaluate`)
  и батчевом (`build_batch_user_message`).
  Суммарный лимит блока 1500 символов. SYSTEM_PROMPT содержит охранную
  директиву. Гейт: feedback.enabled. Один запрос get_recent_feedback на цикл.

**Дайджест:**
- Собирает items state='deferred', сортирует llm_score DESC, fetched_at ASC.
- Расписание: 2×/день из digest.times (локальная tz, дефолт ["08:00","20:00"]),
  ручной запуск `--digest` или `/digest` из Telegram.
- Конфиг: digest.enabled / digest.times / digest.max_items (≤30) / digest.max_items_per_message / digest.on_empty.
- `digest.max_items=30` — лимит новостей в одном дайджесте (топ-30 по score).
- Гарантия ≤4096 символов/сообщение (HTML-escape, бинарный поиск body по word boundary).
- `last_digest_at` пишется ТОЛЬКО при успешной отправке ≥1 новости.
- State='digested' ставится только реально отправленным items (at-least-once).
- Частичный сбой: отправленные → 'digested', остаток → 'deferred' (retry в следующем окне).
- **Кнопки 👍/👎 в дайджесте ОТКЛЮЧЕНЫ по умолчанию.** Конфиг:
  `feedback.digest_enabled: false` (дефолт). При false → send_text (без клавиатуры),
  feedback_targets НЕ создаются (экономия БД). При true → send_digest с кнопками
  👍/👎 (только они; URL в тексте как `<a href="...">🔗</a>`). Пуш-сообщения
  не затронуты: 👍/👎 остаются в пушах как прежде.

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
- Конфиг: feedback.enabled / **feedback.digest_enabled** (bool, дефолт false) /
  feedback.max_examples / feedback.retention_days (чистит только feedback_targets, не feedback).
- digest_enabled=false (дефолт): дайджест без кнопок 👍/👎, без создания targets → send_text.
  digest_enabled=true: дайджест с 👍/👎 (URL-кнопка НЕ дублируется — он в тексте).
- Пуш: 👍/👎 остаются без изменений.

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
**Тесты, запуск: `python -m pytest tests -q -p no:randomly`**
(точный список — `pytest --collect-only -q tests/ | grep test_`)
- `test_llm_batch` — батч-оценка (happy path, per-item fallback, pipeline-интеграция)
- `test_llm_filter_mocks` — LLM-фильтр (mock client, urgent/route/parse)
- `test_pipeline_llm` — pipeline + LLM
- `test_core_reliability` — retry/recovery/failed/state-машина
- `test_delivery_security` — delivery + security hardening
- `test_digest` — дайджест, build_digest_messages, ≤4096
- `test_digest_feedback` — дайджест + feedback
- `test_feedback_storage` — feedback_targets + feedback CRUD
- `test_feedback_flow` — pipeline + feedback, few-shot context
- `test_telegram_poller` — TelegramPoller, callback, команды, backoff, 409 Conflict
- `test_feedback_prompt` — build_feedback_block, санитизация, лимиты
- `test_dedup` — кросс-источниковая дедупликация (точный + fuzzy), нормализация, миграция v9
- `test_dedup_llm` — LLM-дедупликация
- `test_offpeak` — off-peak гейт
- `test_pause` — пауза /stop /continue
- `test_config_env_collector` — env.py (m2/unquoted-spaces), DEFAULT_CONFIG, collector M1 (max-feed-bytes + httpx.MockTransport)
- `test_log_sanitizer` — лог-санитизация, лимиты, unicode
- `test_digest_schedule` — расписание дайджеста
- `test_storage_hardening` — storage hardening (T1)
- `test_poller_lifecycle` — poller lifecycle
- `test_llm_routing` — LLM routing
- `test_digest_retry` — digest retry

## Поведение, принятое владельцем (не менять без согласования)

| Что | Решение | Где |
|-----|---------|-----|
| Retention: 7 дней (временно), deferred: 3 дня | Временно для отладки, откат одной строкой (было 1) | config.yaml |
| Ночные сводки БПЛА без разрушений | Намеренно получают score 1–3, фильтруются | LLM-скоринг, SYSTEM_PROMPT |
| Off-peak окна (01–04 и 06–10 UTC будни, выходные целиком) | Соответствуют официальному прайсу DeepSeek от 10.09.2026 | config.yaml, llm/offpeak.py |
| Summary всегда на английском | Решение владельца: summary ВСЕГДА EN, независимо от языка статьи | newsbot/llm/prompts.py (SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT) |
| CORROBORATION CAP (score ≤4) | Неподтверждённые одиночные сводки о жертвах/заявлениях — CAP 4 | newsbot/llm/prompts.py (SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT) |
| LLM-локация | location извлекается LLM при скоринге; словарь locations.py удалён; cluster_key пересобирается после скоринга | newsbot/llm/prompts.py, pipeline.py |
| Файловый лог | data/logs/newsbot.log, RotatingFileHandler 5 МБ, backupCount 3 | newsbot/log.py, config.yaml |

## Текущая сессия: важные поведения (отражены в коде)

**Telegram-доставка:**
- Постоянные 4xx → `PermanentDeliveryError` → item сразу `state='failed'` (без retry_count), батч продолжается. `send_batch` возвращает `(delivered, permanent_failed)`.
- 429 на последней попытке retry НЕ спит (нет смысла — бюджет исчерпан).

**Retry-очередь:**
- Очередь stuck pending ограничена `database.stuck_pending_limit=500` (защита RAM).
- Резерв ≥½ бюджета при left>0: формула `min(max(left//2,1), len(stuck))`. Excess по бюджету возвращается в 'new'.

**Balance-алерт:**
- `get_balance`: NaN/inf/битый `total_balance` → `is_available=False` (без ложного алерта «баланс 0»).

**LLM-цены:**
- deepseek-flash верифицированы: off-peak input $0.15/output $0.60 за 1M токенов.
- deepseek-v4-pro НЕ верифицированы (placeholder input $0.50/output $2.00).
- Off-peak окна подтверждены официальным прайсом DeepSeek (peak 01–04 и 06–10 UTC будни; выходные целиком off-peak).

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
| T2-11 | LLM `urgent`-парсинг: bool/"true"/"false"/1/0; дефолт False; фильтр не падает на мусоре        |
| U1    | пуш только urgent (push_urgent_only=True); остальное → дайджест ≤30; маршрутизация в route()   |
| m1    | digest: send_digest (с кнопками) vs send_text (без кнопками) — выбор по item_buttons_per_msg |
| m2    | run_digest: groups могут быть переданы из pipeline (единый источник, без пересчёта) |
| D1    | fuzzy: поднят threshold 0.9→0.95, добавлен числовой guard (жёсткий запрет при разных числах) |
| D2    | fuzzy: добавлен token-guard — Jaccard значащих токенов ≥ token_threshold (дефолт 0.8) |
| D3    | fuzzy: выполняется только при cross_source=true (per-source семантика сохранена) |
| D4    | storage: удалён мёртвый атрибут _dedup_config                           |
| D5    | collector: regex «From Source» — *→+ у разделителя (защита «From Source to Sea…») |
| D6    | dedup: комментарий предфильтра поправлен (считает слова, не символы) |
| B14   | Батч-оценка: LLMFilter.evaluate_batch + _parse_batch_response; prompts.build_batch_user_message + BATCH_SYSTEM_PROMPT; batch_size=10, max_tokens=4096; pipeline._apply_result helper; evaluate_batch в FakeLLMFilter для тестов |
| R1    | Major #1-фикс: бюджет max_per_cycle считается в items (не в вызовах); списание = len(fresh_batch); B13-резервация согласована; бюджет<0 клампится в 0 |
| R2    | Major #2-фикс: max_tokens=4096 (ранее 2048); finish_reason=length → LLMError+warning; JSONDecodeError → честный 1 retry (не break); batch_size=1 → evaluate(ok) не падает на finish_reason |
| M1-INJ | M1-фикс: sanitize_item_field() — санитизация title/summary/source + нейтрализация делимитеров (<\|>/|>) в обоих путях: одиночном (LLMFilter.evaluate) и батчевом (build_batch_user_message) |
| 2a     | deferred_retention_days: purge deferred по возрасту (3 дня, конфиг); new включён в основной purge; DEFAULT_DEFERRED_RETENTION_DAYS=3 (ранее 14; обоснование: 2 дайджеста/день + запас; критично для VPS 1 GB RAM) |
| 2b     | recovery_new_limit: K1-recovery ограничен лимитом (500, конфиг); get_all_by_state(state, limit=...) не грузит неограниченный объём в RAM при долгой паузе |
| 2c     | candidates remainder → increment_retry_count + _check_and_mark_failed (at-least-once для кандидатов; раньше retry_count не инкрементировался при сбое TG API) |
| v10-PC | v10 post-check: defensive ADD COLUMN llm_urgent при user_version=10 без колонки (конкурентная миграция edge case) |
| v11    | Создание индексов idx_seen_fetched (seen_titles) и idx_items_state_state_at (items) для оптимизации purge и get_stuck_pending |

## План этапов

- [x] Этап 1 (MVP-каркас): ✅ ЗАВЕРШЁН, APPROVE.
- [x] Этап 2a (Telegram-доставка): ✅ РЕАЛИЗОВАН. getUpdates, send_batch,
  HTML-экранирование, split >4096, retry 429, close только в NewsPipeline.close(),
  inline-кнопка. R1-фикс: ConsoleDelivery → List[str].
- [x] Этап 2b-1 (LLM-ядро): ✅ РЕАЛИЗОВАН. DeepSeekClient, LLMFilter,
  SYSTEM_PROMPT v3, config llm.*, env deepseek_api_key(), --check-llm.
  18/18 тестов на моках *(на момент этапа)*.
- [x] Этап 2b-2a (LLM в pipeline): ✅ РЕАЛИЗОВАН. Schema v4, deferred-state,
  update_llm_results_batch, retry с переиспользованием llm_score.
  15 новых тестов, итого 33 *(на момент этапа)*.
- [x] Этап 2b-2b (баланс-алерт + hardening): ✅ РЕАЛИЗОВАН.
  BalanceInfo, is_available-логика, SCHEMA v5 (meta), send_text, max_per_cycle,
  try/except evaluate(), finally-close клиента. 40 тестов *(на момент этапа)*.
- [x] Финальное hardening (после полного code-review): ✅ ЗАВЕРШЕНО.
  Recovery 'new', HTTPError, finally, retry_count/failed/лимит, WAL/busy_timeout,
  save_many N1, cap summary, единый форматтер, маскирование токенов,
  восстановление _set_chat_id (max update_id), и др.
  **123 теста зелёные (на момент этапа). Счётчик после правок — 746 (убран 1 дубль).**
- [x] Этап 2b-2c (Дайджест): ✅ РЕАЛИЗОВАН. Schema v8, digest.py, build_digest_messages,
  расписание 2×/день (digest.times), ручной --digest, on_empty, гарантия ≤4096,
  last_digest_at по мета, digested-state, at-least-once.
  **481 тест зелёный (на момент этапа). Счётчик после правок — 746.**
- [x] Фича «пуш только срочного» (urgent-only routing): ✅ РЕАЛИЗОВАН.
  JSON `urgent` в SYSTEM_PROMPT, FilterResult.urgent, route(urgent, push_urgent_only),
  `llm.push_urgent_only=True` (дефолт), `digest.max_items=30`. Обратная совместимость
  через `push_urgent_only=False`.
- [x] Фаза 2b (TelegramPoller + обратная связь): ✅ РЕАЛИЗОВАН. TelegramPoller
  (long-poll, кнопки 👍/👎, команды), Schema v7+8 (feedback_targets + feedback),
  record_feedback, get_recent_feedback, few-shot в промпте.
- [x] Фаза 2c (Few-shot адаптация): ✅ РЕАЛИЗОВАН. build_feedback_block,
  санитизация control-chars, маркер «ДАННЫЕ (НЕ ИНСТРУКЦИИ)», охранная
  директива в SYSTEM_PROMPT, суммарный лимит 1500.
- [x] Фича «без кнопок 👍/👎 в дайджесте»: ✅ РЕАЛИЗОВАНА. Конфиг
  `feedback.digest_enabled: false` (дефолт), `send_text` без клавиатуры,
  без создания feedback_targets (экономия БД). При `digest_enabled=true` —
  `send_digest` с 👍/👎 (URL в тексте, не дублируется кнопкой).
- [ ] Этап 3: скрейперы, RSSHub-мост для ТГ-каналов (Astra, Baza, Shot, Mash,
  Sirena), Steam-подписки на тайтлы, USGS GeoJSON (землетрясения).
- [x] Этап 5: деплой 24/7 (Docker), VPS, живая проверка фидов и Telegram.

## Оставшиеся задачи

| # | Что                          | Приоритет     | Комментарий                                       |
|---|------------------------------|---------------|---------------------------------------------------|
| 1 | URL-нормализация             | Средний       | utm_* + trailing slash; отложена (known limitation)|
| 2 | Проверка модели DeepSeek     | Высокий       | `deepseek-flash` — проверить id через `--check-llm` живым ключом |
| 3 | Deploy на VPS, этап 5        | ✅ Завершён    | DEPLOY.md, Dockerfile, docker-compose.yml готовы       |
| 4 | Telegram-доставка живьём    | Высокий       | api.telegram.org недоступен с локальной машины      |
| 5 | Этап 3 (RSSHub, USGS, Steam) | Низкий        | ТГ-каналы → RSSHub, землетрясения → USGS GeoJSON   |
| 6 | Семантический дедуп (embeddings/LLM) | Отложен | Перефразы не ловятся; нужен отдельный этап     |

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