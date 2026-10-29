# newsbot — персональный новостной бот

> Конвейер: сбор RSS → дедупликация → LLM-фильтр (DeepSeek) → доставка в Telegram (httpx + Bot API).
> Обратная связь: 👍/👎 → few-shot адаптация фильтра.
> Дайджест: отложенные новости (score 5–6) отправляются по расписанию 2×/день.
> При падении LLM — fallback: все новости доставляются непрофильтрованными.

## Быстрый старт

```bash
# 0. Клонирование репозитория
git clone https://github.com/VladSmr/newsbot.git
cd newsbot
# На GitHub основная ветка — main (если git checkout не переключил автоматически):
git checkout main

# 1. Виртуальное окружение (Python 3.8+, реально используется 3.12)
python -m venv venv
source venv/bin/activate      # Linux/macOS
.\venv\Scripts\activate       # Windows

# 2. Зависимости
pip install -e .
#  или  pip install -r requirements.txt

# 3. Токены — в .env (рядом с config.yaml)
#     .env уже есть в .gitignore
#  TELEGRAM_BOT_TOKEN  — от @BotFather
#  TELEGRAM_CHAT_ID   — от --set-chat-id
#  DEEPSEEK_API_KEY    — от platform.deepseek.com

# 4. Привязка chat_id (владелец нажимает /start у бота)
python -m newsbot --set-chat-id

# 5. Проверка токена и LLM
python -m newsbot --check-telegram
python -m newsbot --check-llm

# 6. Один цикл (сухой прогон)
python -m newsbot --once

# 7. Демон: бесконечный цикл + TelegramPoller (кнопки 👍/👎, команды)
python -m newsbot

# 8. Дайджест вручную
python -m newsbot --digest
```

## Структура проекта

```
newsbot/                  ← проектный корень
├── newsbot/              ← python-пакет (pip install -e .)
│   ├── __init__.py
│   ├── __main__.py       # python -m newsbot [--once|--digest|--check-*]
│   ├── config.py         # загрузка config.yaml + .env
│   ├── env.py            # парсер .env
│   ├── log.py            # настройка logging
│   ├── collector.py      # feedparser + httpx, FeedResult
│   ├── storage.py        # SQLite: items + seen_titles + meta + feedback, state-машина
│   ├── dedup.py          # filter_new()
│   ├── pipeline.py       # конвейер, state-машина, retry, дайджест
│   ├── telegram_io.py    # TelegramPoller (long-poll, кнопки, команды)
│   ├── digest.py         # формирование и отправка дайджеста
│   ├── delivery/
│   │   ├── __init__.py   # Delivery Protocol (structural typing)
│   │   ├── console.py    # ConsoleDelivery (fallback)
│   │   └── telegram.py   # TelegramDelivery (Bot API, HTML, split >4096)
│   └── llm/
│       ├── __init__.py   # LLMFilter, FilterResult, DeepSeekClient, route
│       ├── client.py     # DeepSeekClient (chat_json, get_balance, retry)
│       ├── filter.py     # LLMFilter, FilterResult, route()
│       └── prompts.py    # SYSTEM_PROMPT (источник истины) + build_feedback_block
├── config.yaml           # 30 фидов, интервалы, путь к БД, llm.*, digest.*, feedback.*
├── .env                  # TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DEEPSEEK_API_KEY
├── requirements.txt
├── setup.py
├── MANIFEST.in
├── tests/                # тесты: запуск `python -m pytest tests -q`
                      # список файлов: `pytest --collect-only -q tests/ | grep test_`
└── data/                 # newsbot.db, newsbot.db-wal, newsbot.db-shm
```

**Путь к БД:** `data/newsbot.db` относительно проектного корня — независимо от CWD.

## Конвейер (pipeline)

```
Collector (feedparser + httpx, 30 фидов)
       ↓
Dedup (URL точный + title+source нормализованный, TTL 30 дн.)
       ↓
save_many() — INSERT OR IGNORE, state='new', retry_count=0
       ↓
[recovery K1] — подхват 'new'-items из прошлого цикла (at-least-once)
       ↓
[recovery C2] — терминализация зависших 'pending' с retry_count >= лимита
       ↓
Few-shot: загрузка последних 👍/👎 из БД (feedback.enabled)
       ↓
LLM-фильтр (DeepSeek, max_per_cycle=250 на цикл, batch_size=10, wall-clock слоты :30)
  retry-items с готовым llm_score НЕ тратят бюджет
  Батч-оценка: ≤10 новостей одним вызовом (экономия system prompt)
  Few-shot блок «ДАННЫЕ (НЕ ИНСТРУКЦИИ)» подставляется в промпт
       ↓
set_state('pending') перед send_batch (защита от double-send)
       ↓
Delivery.send_batch() → set_state('delivered')
       ↓
Retry: get_stuck_pending() — 'pending' старше retry_threshold_sec (60 с)
       ↓
Purge: seen_titles (TTL) + items (delivered/skipped/failed/digested, retention_days)
```

## State-машина (7 состояний)

| State       | Смысл                                        | Retry?                        |
|-------------|----------------------------------------------|-------------------------------|
| `new`       | Обнаружен, сохранён, ждёт обработки           | Подхватывается в след. цикле  |
| `pending`   | Обработка начата (set_state перед send_batch)| Да — через `get_stuck_pending` |
| `delivered` | Успешно доставлен (пуш)                      | Нет                           |
| `skipped`   | LLM отфильтровал (score < digest_threshold)  | Нет                           |
| `deferred`  | Score 5–6, отложен на дайджест               | Нет; чистится по deferred_retention_days (3 дня) |
| `failed`    | Терминальный; retry_count исчерпан           | Нет (purge по retention_days)   |
| `digested`  | Терминальный; отправлен в дайджесте          | Нет (purge по retention_days)   |

**Retry-механика:**
- `database.max_retries` (по умолч. 8) — лимит попыток.
- При исчерпании → `failed` (терминальный).
- Recovery на старте цикла: зависшие `pending` с `retry_count >= max_retries` получают `failed`.
- При падении delivery: items остаются `pending`, retry в следующем цикле.

**Purge:** `delivered`, `skipped`, `failed`, `digested` старше `retention_days` (1 день);
`new` удаляется при возрасте ≥ max(`retention_days`, 7) — минимум 7 дней;
`deferred` удаляются по отдельному окну `deferred_retention_days` (3 дня по умолчанию)
— старые неотправленные отложенные новости больше не копятся вечно.

## CLI-команды

```bash
python -m newsbot                        # демон, бесконечный цикл + TelegramPoller
python -m newsbot --once                 # один цикл и выход
python -m newsbot --digest               # дайджест отложенных новостей и выход
python -m newsbot --check-telegram       # getMe → username бота
python -m newsbot --set-chat-id          # получить chat_id через getUpdates
python -m newsbot --set-chat-id --bind OWNER_ID  # привязать конкретный chat_id
python -m newsbot --check-llm            # баланс + тестовый chat_json
python -m newsbot --version              # показать версию приложения
```

**TelegramPoller** запускается только в демон-режиме (`python -m newsbot`).
Не запускается при `--once` / `--digest` / `--check-*` (конфликт getUpdates).

## LLM-фильтр (DeepSeek)

**Настройка в config.yaml:**
```yaml
llm:
  enabled: true
  provider: deepseek
  model: deepseek-flash           # ⚠️ проверить id живым ключом через --check-llm
                                  #   deepseek-v4-pro — более способная/дорогая альтернатива (одной строкой)
  base_url: https://api.deepseek.com
  push_urgent_only: true         # пуш ТОЛЬКО срочных (urgent=true, score>=7); остальное → дайджест
                                  # ⚠️ при отсутствии ключа — WARNING + дефолт True (urgent-only)!
                                  #    legacy-конфиг без ключа молча переключится в urgent-only.
                                  #    Явно задайте push_urgent_only: false для прежнего поведения.
  push_threshold: 7              # вторичный гейт: score >= 7 + urgent=True → пуш
  digest_threshold: 5             # score 5–6 → deferred (дайджест); < 5 → skipped
  max_retries: 3                  # retry при 429/5xx
  timeout_sec: 60
  balance:
    enabled: true
    threshold: 2.0               # алерт при балансе ниже порога (USD)
    check_interval_hours: 24
  max_per_cycle: 250             # макс. items (оценок) на цикл (12 слотов × 250 = 3000/сут)
  max_tokens: 4608               # лимит completion tokens (≈4600 для батча из 10)
  batch_size: 10                 # ≤10 новостей одним LLM-вызовом (1=поштучно)
  # off_peak_only: false         # по умолчанию; раскомментируйте для экономии (×0.5)
```

**Маршрутизация** (JSON от LLM: `{"score": 0–10, "summary", "reason", "urgent": true|false}`):

| Условие                                   | Результат        |
|-------------------------------------------|------------------|
| `urgent=true` AND `score≥7`              | пуш (delivered)  |
| `score 5–6` (даже если urgent, но score<7)| дайджест (deferred) |
| `score<5`                                | skipped          |

`push_urgent_only=false` (обратная совместимость): `score≥7` → пуш,
игнорирует `urgent`.

**Retry-семантика:** `llm_urgent` персистентен (БД-схема v11, колонка `items.llm_urgent` INTEGER 0/1).
При retry: срочность восстанавливается из БД → `FilterResult(..., urgent=bool(llm_urgent))`.
Legacy-строки (до v10, `llm_urgent=NULL`) получают `urgent=False` (безопасный дефолт).

**Fallback (ok=False):** при сбое LLM:
- `push_urgent_only=True` → deferred (дайджест) — защита от спама пушами;
- `push_urgent_only=False` → пуш (legacy-поведение).

**⚠️ Проверка модели:** `python -m newsbot --check-llm`
Если API вернёт 4xx/404 на невалидную модель — фильтр НЕ сработает,
все новости пойдут fallback'ом (push при `push_urgent_only=False`, иначе deferred).
Проверьте имя модели в `platform.deepseek.com/api_keys` и обновите `llm.model`.

**System prompt** — константа `SYSTEM_PROMPT` в `newsbot/llm/prompts.py`.
Её содержимое — источник истины. Поле `llm.system_prompt` в config.yaml
пустое (или отсутствует) — это штатный оверрайд.

**Fallback (LLM error / missing key / invalid key):** все новости
доставляются непрофильтрованными:
- `push_urgent_only=True` → deferred (дайджест);
- `push_urgent_only=False` → пуш;
- кнопки 👍/👎 не добавляются (нет `feedback_target_id`).

**Батч-оценка (`llm.batch_size`, по умолчанию 10):**

≤10 новостей оцениваются одним LLM-вызовом. System prompt платится **раз на батч**, а не на каждую новость — экономия input-токенов ≈ N×(system_prompt_size). При `batch_size=1` поведение идентично поштучному (`evaluate()`).

Формат батч-ответа LLM: `{"results": [{"id": 1, "score": 0-10, "urgent": bool, "summary": "...", "reason": "..."}, ...]}`. Per-item валидация: пропущенный/битый id или невалидные поля → только этот item получает `ok=False` (fallback), остальные обрабатываются normally. Полный сбой LLM (`LLMError`) → все items `ok=False` (fallback).

Retry-items с сохранённым `llm_score` **не попадают в батч** — они синтезируют `FilterResult` из БД и не тратят бюджет.

`llm.max_tokens` (по умолчанию 4608): лимит completion tokens.
  - Батч из 10 items ≈ 4000 токенов ответа + запас → 4608 покрывает типичный случай.
  - При усечении (`finish_reason="length"`) LLM поднимает `LLMError` с warning-логом,
    батч получает fallback (per-item: ok=False → доставка непрофильтрованной).
  - Конфиг позволяет переопределить (`batch_size=5`, и т.п.).

**`llm.off_peak_only`** (по умолчанию `false`): ограничить LLM-вызовы
непиковыми часами DeepSeek (там цена ×0.5). В пик новые новости
собираются, остаются необработанными (`new`; подхватятся в следующем
off-peak цикле). Ручной запуск `--once` тоже гейтится (консистентно).

| Период                        | UTC          | MSK (UTC+3)           |
|-------------------------------|--------------|-----------------------|
| **Пик** (включить нельзя)     | 01:00–03:59  | 04:00–06:59           |
|                               | 06:00–09:59  | 09:00–12:59           |
| **Off-peak** (можно)          | всё остальное время + выходные |

> ⚠️ Китайские государственные праздники консервативно считаются пиком
> (не учитываются — модель будет выключена в праздник вместо работы со скидкой).

## Telegram-доставка

**Выбор delivery:** если `TELEGRAM_BOT_TOKEN` и `TELEGRAM_CHAT_ID` заданы →
`TelegramDelivery`, иначе → `ConsoleDelivery`.

**Привязка chat_id (однократно):**
```
1. Откройте чат с ботом в Telegram
2. Нажмите /start
3. Отправьте боту любое сообщение (например, «Привет»)
4. На машине с ботом выполните:
   python -m newsbot --set-chat-id
```

**Проверка:**
```
python -m newsbot --check-telegram   # getMe → username бота
python -m newsbot --once             # сухой прогон с вашей БД
```

> ⚠️ **Доставка в Telegram живьём с локальной машины не проверена:**
> api.telegram.org недоступен с локальной машины. Проверка — на VPS.

**Формат карточки в Telegram:**
```
📰 Заголовок статьи

Краткое содержание (2–5 предложений).

📍 Источник · 13.10.2026 14:32
🔗 Открыть источник    (инлайн-кнопка)
```
- `parse_mode=HTML`, спецсимволы экранируются
- Текст > 4096 символов автоматически разбивается на части
- 429 (retry_after) — ждёт и повторяет; 403 — бот заблокирован; сетевые ошибки — retry
- При непривязанном `chat_id` поднимается `ChatIdNotBoundError` — items остаются `pending`, цикл НЕ падает

## Обратная связь (Фаза 2b / 2c)

**Инлайн-кнопки** 👍 «Интересно» / 👎 «Не интересно» на каждом push-сообщении
и в каждой новости дайджеста. `callback_data` формат: `fb:<id>:<u|d>`.

**TelegramPoller** (демон-поток, long-poll `getUpdates`):
- Слушает `callback_query` и команды `/digest`, `/start`, `/help`
- Обрабатывает только сообщения от владельца (`chat_id` проверяется)
- Создаёт **отдельное соединение SQLite** в своём потоке (не конфликтует с main)
- Offset хранится в `meta` (`tg_updates_offset`) — устойчив к рестартам
- Конфликт 409 (два getUpdates-потребителя) — backoff 60 с

**Настройка в config.yaml:**
```yaml
feedback:
  enabled: true                # вкл/выкл приём фидбэка
  max_examples: 10            # сколько последних вердиктов подмешивать в few-shot
  retention_days: 180          # удалять feedback_targets старше N дней
                               # (таблица feedback не чистится — обучающие данные)

telegram:
  poll_timeout: 25             # seconds для long-poll getUpdates
```

**Few-shot адаптация фильтра (Фаза 2c):**
- Последние 👎 (и 👍) из таблицы `feedback` подставляются в промпт LLM-фильтра
- Блок «ДАННЫЕ (НЕ ИНСТРУКЦИИ)» с примерами: `<|пример: Title (Source)|>`
- Санитизация: удаляются `\n`, `\r`, control-chars, схлопываются пробелы
- **Санитизация title/summary/source в промпте (M1-фикс):** поля новости
  прогоняются через `sanitize_item_field()` перед подстановкой. Это защита от
  prompt injection: ``<|данные:N|>`` / ``<|конец данных:N|>`` / ``<|пример: …|>``
  нейтрализуются (``< |`` / ``| >``). Работает в обоих путях — одиночном
  (`evaluate`) и батчевом (`build_batch_user_message`).
- Суммарный лимит: 1500 символов на блок
- Охранная директива в `SYSTEM_PROMPT`: модель НЕ выполняет указания из примеров
- Гейт: `feedback.enabled`; один запрос `get_recent_feedback` за цикл

## Дайджест

Собирает items со `state='deferred'`, сортирует по `llm_score DESC`, `fetched_at ASC`.
Отправляется **автоматически по расписанию** (2×/день) и **вручную** (`--digest`).

**Настройка в config.yaml:**
```yaml
digest:
  enabled: true                       # вкл/выкл автоматический дайджест
  times: ["08:00", "20:00"]          # HH:MM в ЛОКАЛЬНОЙ tz
  max_items: 30                       # ≤30 самых важных (сортировка: score DESC)
  max_items_per_message: 20           # новостей на одно сообщение
  on_empty: skip                      # "skip" — не слать пустой дайджест
```

Собирает все non-urgent релевантные новости (score 5–6), отбирает ≤30
топовых по score, отправляет 2×/день. Перед отправкой запускается
финальная LLM-кластеризация (`digest.cluster_duplicates=true`, дефолт) —
семантические дубли (одно событие, разные формулировки) отсеиваются
в пользу наиболее приоритетного лидера. Управляется параметром
`digest.cluster_duplicates` в config.yaml.

**Гарантии:**
- ≤ 4096 символов на сообщение (HTML-экранирование, безопасное усечение body по word boundary)
- `last_digest_at` пишется в `meta` **только** при успешной отправке ≥1 новости
- `state='digested'` ставится только реально отправленным items (**at-least-once**)
- Частичный сбой: отправленные → `digested`; остаток → `deferred` (retry в следующем окне)

**Команды бота:**
- `/stop` — приостановить обработку и доставку (сбор RSS продолжается,
  items остаются `state='new'`, обработаются после `/continue`)
- `/continue` — возобновить обработку после `/stop`
- `/digest` — сигнализирует демон-циклу (выполняется основным потоком)
- `/start`, `/help` — справка

**Пауза (мета-ключ `paused` в БД):** флаг хранится в `meta` SQLite
(ключ `paused`="1"/"0"), переживает рестарт контейнера. При паузе:
RSS-сбор → `state='new'` (продолжается) → LLM-фильтр, доставка,
авто-дайджест и алерт баланса — пропускаются. Mid-batch пауза
останавливает обработку, оставшиеся items → `state='pending'`
(retry в следующем цикле, at-least-once).

## Дедупликация

**Три уровня (уровни 1–2 конфигурируемые, уровень 3 всегда включён):**

**Уровень 1 — глобальный точный:** `title_hash` БЕЗ `source` в пределах `window_days`.
Активируется: `dedup.cross_source=true` (дефолт True). Окно: `dedup.window_days`
(дефолт 3 дня). Первая копия идёт как обычно, последующие — `skipped`.

**Уровень 2 — нечёткий (fuzzy):** `difflib.SequenceMatcher` с предфильтром
(токен-сеты / длина) и лимитом кандидатов — не O(N²). Активируется:
`dedup.fuzzy=true` (дефолт True), **только при `cross_source=true`** (кросс-источниковый
контекст). Порог: `dedup.threshold` (дефолт 0.95). Дубли → `skipped`.

**Защита от ложных склеек** — три условия И-вместе:
  (a) `ratio ≥ threshold` — SequenceMatcher ratio
  (b) **числовой guard** — мультимножества (кратности учитываются) `\d+(?:[.,]\d+)?` совпадают (жёсткий запрет)
  (c) **token-guard** — Jaccard значащих токенов ≥ `token_threshold` (дефолт 0.8)
      (слова ≥4 букв., без стоп-слов и чисел)

`дубль ⟺ (a) И (b) И (c)`

Предфильтр: `MAX_LEN_DIFF=15` слов, `MIN_TOKEN_INTERSECTION=0.3`,
`MAX_FUZZY_CANDIDATES=200`. Новые параметры: `dedup.threshold` (0.95),
`dedup.token_threshold` (0.8). При `cross_source=false` fuzzy игнорируется (per-source
поведение сохранено).

**Всегда:**
— **URL** — точный, живёт в пределах жизни записи в `items` (URL-дедуп в таблице
  — `INSERT OR IGNORE`, UNIQUE INDEX `idx_items_url_norm` по нормализованному URL). Purge: терминальные — 1 день,
  `deferred` — 3 дня, `new` — минимум 7 дней. Отдельного permanent-хранилища URL нет
  (заголовки — отдельная история: `seen_titles` TTL 30 дней).
- **Заголовок** — нормализованный (strip, collapse spaces, lowercase, remove punctuation,
  срез префиксов `Reuters:`, `Bloomberg:`, `Источник:` и т.п.) + `source`, TTL 30 дней.
  Заголовки короче 3 символов НЕ пишутся в `seen_titles` (MIN_TITLE_LEN=3).
- **cluster_key** — доменный 2-часовой дедуп: ключ = `{location}|{stem_hash}|{pub_day}`
  (fallback: `nostem:{md5(title)}`); `tags` (json-список `<category>` из фида) в ключ
  НЕ входит. Если в пределах 2 часов уже была новость с тем же `cluster_key`, она
  помечается `skipped`. Всегда включён, не конфигурируется.

**Уровень 3 — LLM-дедуп (семантическая группировка «одно событие»):**
`llm.llm_dedup: true` (дефолт True). Отдельный лёгкий вызов (только заголовки) **ДО**
LLM-скоринга. Критерий: «одно и то же событие, разные слова, язык, формулировки чисел».
Примеры: RU — «сбили 80 дронов» ↔ «сбили 80 БПЛА»; EN — «34 children killed» ↔ «dozens killed».
Промпт: специализированный (`DEDUP_SYSTEM_PROMPT`), делимитеры `<|данные:N|>` защищают от injection.
Чанкование по 40 кандидатов; `recent` (до 20 недавних новостей из БД) подмешивается в каждый чанк.
При **сбое LLM** — безопасный пропуск (пустое множество, ни один дубль не проходит).
Битые/несуществующие индексы в ответе игнорируются. При `<2` кандидатов вызов не делается.
Дубли → `skipped` (URL-дедуп в следующих циклах) + исключаются из дальнейшей обработки.

## Конфигурация

| Ключ                          | Описание                            | По умолчанию          |
|-------------------------------|-------------------------------------|-----------------------|
| `poll_interval_minutes`       | Grace-период для published-фильтра (мин.); в репозиторном `config.yaml`=120 (соответствует 2-ч циклу); дефолт в коде=15 | 15 (код), 120 (config.yaml) |
| `database.path`               | Путь к SQLite (от каталога newsbot)  | `data/newsbot.db`     |
| `database.retention_days`     | Хранить delivered/skipped/failed/digested и new старше 7 дней | 1          |
| `database.retry_threshold_sec` | pending считается зависшим через    | 60                    |
| `database.max_retries`        | Лимит retry попыток → state=failed   | 8                     |
| `database.deferred_retention_days` | Удалять старые deferred-items (не попали в digest) | 3               |
| `database.stuck_pending_limit` | Макс. 'pending'-items, загружаемых из retry-очереди за цикл (защита RAM) | 500 |
| `database.recovery_new_limit` | Макс. 'new'-items, загружаемых в RAM за цикл (K1-recovery) | 500          |
| `llm.max_per_cycle`           | Макс. LLM-оценок (items) на цикл; 150 в коде, 250 в config.yaml (12 слотов × 250 = 3000) | 250             |
| `llm.batch_size`             | ≤N новостей одним LLM-вызовом (1=поштучно) | 10              |
| `llm.max_tokens`            | Лимит completion tokens (≈4600 для батча 10) | 4608           |
| `llm.off_peak_only`           | LLM только в off-peak часы DeepSeek   | false                 |
| `llm.llm_dedup`               | Семантическая дедупликация LLM (группировка «одно событие») | true          |
| `dedup.llm_recent_limit`      | Сколько недавних заголовков подмешивать в LLM-дедуп | 20         |
| `logging.level`               | DEBUG / INFO / WARNING               | INFO                  |
| `telegram.token`              | Bot token от @BotFather              | из `.env`             |
| `telegram.chat_id`            | Привязанный chat_id владельца        | из `.env`             |
| `telegram.poll_timeout`        | seconds для long-poll               | 25                    |
| `llm.model`                   | Модель DeepSeek                      | `deepseek-flash` (pro — альтернатива) |
| `digest.enabled`              | Автоматический дайджест              | true                  |
| `digest.times`                | HH:MM в локальной tz                 | `["08:00","20:00"]`   |
| `digest.cluster_duplicates`   | LLM-кластеризация deferred перед отправкой | true           |
| `digest.max_age_hours`        | Выбирать deferred-items не старше N часов | 24               |
| `digest.bootstrap_window_hours`| Окно начального наполнения дайджеста | 24                   |
| `digest.max_per_location`     | Макс. новостей на локацию/день в дайджесте | 2                |
| `feedback.enabled`            | Приём обратной связи                 | true                  |
| `feedback.digest_enabled`     | 👍/👎 кнопки в дайджесте             | false                 |
| `llm.daily_cost_limit`        | Дневной лимит расходов на LLM (USD; 0=выкл) | 0.50          |
| `llm.pricing`                 | Цены моделей (input/output за 1M токенов) | deepseek-flash: 0.15/0.60 |
| `feedback.max_examples`       | Примеров в few-shot                 | 10                    |
| `dedup.cross_source`          | Глобальный точный дедуп по title_hash (без source) | true          |
| `dedup.fuzzy`                 | Нечёткое сравнение (только при cross_source=true)  | true          |
| `dedup.threshold`             | Порог SequenceMatcher ratio (0..1, clamp автоматический) | 0.95    |
| `dedup.token_threshold`       | Порог Jaccard значащих токенов (0..1, clamp автоматический) | 0.8   |
| `dedup.window_days`           | Окно сравнения для fuzzy и cross_source (дни)     | 3               |

**.env (рядом с config.yaml, НЕ в git):**
```
TELEGRAM_BOT_TOKEN=<токен от @BotFather>
TELEGRAM_CHAT_ID=<привязанный chat_id>
TELEGRAM_OWNER_ID=<ваш числовой id (опционально)>
DEEPSEEK_API_KEY=<ключ от platform.deepseek.com>
```

## Схема БД (v13, идемпотентные миграции v1→v13)

```sql
-- items: state-машина
items: id, source, url, title, summary, published, fetched_at,
       state, state_at, llm_score, llm_summary, llm_reason,
       llm_urgent INTEGER (v10/v11, 0/1), filtered_at, retry_count,
       tags TEXT,        -- v4: список тегов (например "attack","russia","drone")
       cluster_key TEXT,-- v12: ключ группировки для 2-часового доменного дедупа
                        -- формат: {location}|{stem_hash}|{pub_day}
       lang TEXT        -- v13: язык новости (ru/en)

-- seen_titles: title-dedup, TTL 30 дней (v9: + title_normalized для fuzzy)
seen_titles: id, title_hash, source, fetched_at, expires_at, title_normalized

-- meta: ключи-значения (last_digest_at, llm_balance_last_check, …)
meta: key TEXT PRIMARY KEY, value TEXT

-- feedback_targets: новость с кнопкой 👍/👎
feedback_targets: id, url, source, title, summary, llm_score,
                  kind ('push'|'digest'), chat_id, message_id, created_at
                  -- индекс (url, kind)

-- feedback: голос пользователя
feedback: id, target_id UNIQUE, url, source, title, summary, llm_score,
          verdict CHECK('like'|'dislike'), chat_id, message_id,
          created_at, updated_at
          -- индексы: (verdict, created_at), (created_at), (updated_at)
```

## Тесты

```bash
python -m pytest tests -q
```
**Тесты**: запуск `python -m pytest tests -q -p no:randomly`
(точный список файлов — `pytest --collect-only -q tests/ | grep test_`):
`test_llm_batch`, `test_llm_filter_mocks`
(urgent/route/parse), `test_pipeline_llm`, `test_core_reliability`,
`test_delivery_security`, `test_digest`, `test_digest_feedback`,
`test_feedback_storage`, `test_feedback_flow`, `test_telegram_poller`,
`test_feedback_prompt`, `test_dedup`, `test_dedup_llm`, `test_offpeak`,
`test_pause`, `test_config_env_collector`, `test_log_sanitizer`,
`test_digest_schedule`, `test_storage_hardening`, `test_poller_lifecycle`,
`test_llm_routing`, `test_digest_retry`. Запуск: `python -m pytest tests -q -p no:randomly`.

## Известные ограничения

| Что                   | Когда          | Комментарий                                              |
|-----------------------|----------------|----------------------------------------------------------|
| Telegram-доставка     | Проверка на VPS | `api.telegram.org` недоступен с локальной машины         |
| Проверка модели DeepSeek | Сейчас        | deepseek-flash — проверить id живым ключом + `--check-llm` |
| URL-нормализация      | Отложена       | UTM-параметры, trailing slash — не убираются             |
| Этап 3 (RSSHub, USGS, Steam) | Не начат | ТГ-каналы через RSSHub; землетрясения через USGS GeoJSON |
| Этап 5 (Docker / VPS) | ✅ Готово        | `Dockerfile`, `docker-compose.yml`, `DEPLOY.md`          |
| at-least-once (send ↔ set_state) | Сейчас | Окно между send и set_state('delivered'); редкий дубль   |
| Хвост retry-очереди   | Сейчас         | Голодание при stuck > max_per_cycle                      |

## Требования

- Python 3.8+ (протестировано на 3.12)

## Деплой на VPS (Docker)

См. [DEPLOY.md](./DEPLOY.md) — полная инструкция: подготовка сервера,
`.env`, привязка chat_id, прокси для Telegram/DeepSeek, бэкап БД,
TZ и расписание дайджеста, обновление.