# newsbot — персональный новостной бот

> Конвейер: сбор RSS → дедупликация → LLM-фильтр (DeepSeek) → доставка в Telegram.
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

# 1. Виртуальное окружение (Python 3.8+)
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
│   │   ├── __init__.py   # Delivery ABC
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
├── tests/                # 333 теста (9 файлов, запуск: python -m pytest tests -q)
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
LLM-фильтр (DeepSeek, max_per_cycle=150 на цикл)
  retry-items с готовым llm_score НЕ тратят бюджет
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
| `deferred`  | Score 5–6, отложен на дайджест               | Нет; исключён из purge         |
| `failed`    | Терминальный; retry_count исчерпан           | Нет (purge по retention_days)   |
| `digested`  | Терминальный; отправлен в дайджесте          | Нет (purge по retention_days)   |

**Retry-механика:**
- `database.max_retries` (по умолч. 8) — лимит попыток.
- При исчерпании → `failed` (терминальный).
- Recovery на старте цикла: зависшие `pending` с `retry_count >= max_retries` получают `failed`.
- При падении delivery: items остаются `pending`, retry в следующем цикле.

**Purge:** `delivered`, `skipped`, `failed`, `digested` старше `retention_days`.
`deferred` **не удаляются** — ждут дайджест.

## CLI-команды

```bash
python -m newsbot                        # демон, бесконечный цикл + TelegramPoller
python -m newsbot --once                 # один цикл и выход
python -m newsbot --digest               # дайджест отложенных новостей и выход
python -m newsbot --check-telegram       # getMe → username бота
python -m newsbot --set-chat-id          # получить chat_id через getUpdates
python -m newsbot --check-llm            # баланс + тестовый chat_json
```

**TelegramPoller** запускается только в демон-режиме (`python -m newsbot`).
Не запускается при `--once` / `--digest` / `--check-*` (конфликт getUpdates).

## LLM-фильтр (DeepSeek)

**Настройка в config.yaml:**
```yaml
llm:
  enabled: true
  provider: deepseek
  model: deepseek-v4-pro          # ⚠️ проверить id живым ключом через --check-llm
  base_url: https://api.deepseek.com
  push_threshold: 7              # score >= 7 → пуш (delivered)
  digest_threshold: 5             # score 5–6 → deferred (дайджест); < 5 → skipped
  max_retries: 3                  # retry при 429/5xx
  timeout_sec: 60
  balance:
    enabled: true
    threshold: 2.0               # алерт при балансе ниже порога (USD)
    check_interval_hours: 24
  max_per_cycle: 150             # макс. LLM-вызовов на цикл
```

**⚠️ Проверка модели:** `python -m newsbot --check-llm`
Если API вернёт 4xx/404 на невалидную модель — фильтр НЕ сработает,
все новости пойдут fallback'ом (доставка без фильтрации). Проверьте имя
модели в `platform.deepseek.com/api_keys` и обновите `llm.model` в config.yaml.

**System prompt** — константа `SYSTEM_PROMPT` в `newsbot/llm/prompts.py`.
Её содержимое — источник истины. Поле `llm.system_prompt` в config.yaml
пустое (или отсутствует) — это штатный оверрайд.

**Пороги:** score ≥ 7 → пуш; 5–6 → deferred; ≤ 4 → skipped.
Результат сохраняется в БД (llm_score, llm_summary, llm_reason, filtered_at).

**Fallback:** если ключа нет, ключ невалиден или LLM вернул ошибку — все
новости доставляются непрофильтрованными (kind='fallback', кнопки не добавляются).

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
  times: ["09:00", "21:00"]          # HH:MM в ЛОКАЛЬНОЙ tz
  max_items: 100                      # лимит новостей в одном дайджесте
  max_items_per_message: 20           # новостей на одно сообщение
  on_empty: skip                      # "skip" — не слать пустой дайджест
```

**Гарантии:**
- ≤ 4096 символов на сообщение (HTML-экранирование, безопасное усечение body по word boundary)
- `last_digest_at` пишется в `meta` **только** при успешной отправке ≥1 новости
- `state='digested'` ставится только реально отправленным items (**at-least-once**)
- Частичный сбой: отправленные → `digested`; остаток → `deferred` (retry в следующем окне)

**Команды бота:**
- `/digest` — сигнализирует демон-циклу (выполняется основным потоком)
- `/start`, `/help` — справка

## Дедупликация

- **URL** — точный, permanent, все записи (без TTL).
- **Заголовок** — нормализованный (lowercase,strip) + source, TTL 30 дней.
- **NOTE:** нормализация URL (удаление `utm_*`, trailing slash) — **отложена**
  (known limitation; будет реализована отдельно).

## Конфигурация

| Ключ                          | Описание                            | По умолчанию          |
|-------------------------------|-------------------------------------|-----------------------|
| `poll_interval_minutes`       | Пауза между циклами                 | 15                    |
| `database.path`               | Путь к SQLite (от каталога newsbot)  | `data/newsbot.db`     |
| `database.retention_days`     | Хранить delivered/skipped/failed/digested | 90          |
| `database.retry_threshold_sec` | pending считается зависшим через    | 60                    |
| `database.max_retries`        | Лимит retry попыток → state=failed   | 8                     |
| `llm.max_per_cycle`           | Макс. LLM-вызовов на цикл           | 150                   |
| `logging.level`               | DEBUG / INFO / WARNING               | INFO                  |
| `telegram.token`              | Bot token от @BotFather              | из `.env`             |
| `telegram.chat_id`            | Привязанный chat_id владельца        | из `.env`             |
| `telegram.poll_timeout`        | seconds для long-poll               | 25                    |
| `llm.model`                   | Модель DeepSeek                      | `deepseek-v4-pro`     |
| `digest.enabled`              | Автоматический дайджест              | true                  |
| `digest.times`                | HH:MM в локальной tz                 | `["09:00","21:00"]`   |
| `feedback.enabled`            | Приём обратной связи                 | true                  |
| `feedback.max_examples`       | Примеров в few-shot                 | 10                    |

**.env (рядом с config.yaml, НЕ в git):**
```
TELEGRAM_BOT_TOKEN=<токен от @BotFather>
TELEGRAM_CHAT_ID=<привязанный chat_id>
DEEPSEEK_API_KEY=<ключ от platform.deepseek.com>
```

## Схема БД (v8, идемпотентные миграции v1→v8)

```sql
-- items: state-машина
items: id, source, url, title, summary, published, fetched_at,
       state, state_at, llm_score, llm_summary, llm_reason,
       filtered_at, retry_count

-- seen_titles: title-dedup, TTL 30 дней
seen_titles: id, title_hash, source, fetched_at, expires_at

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
**333 теста, 9 файлов:** `test_llm_filter_mocks`, `test_pipeline_llm`,
`test_core_reliability`, `test_delivery_security`, `test_digest`,
`test_feedback_storage`, `test_feedback_flow`, `test_telegram_poller`,
`test_feedback_prompt`. Все зелёные.

## Известные ограничения

| Что                   | Когда          | Комментарий                                              |
|-----------------------|----------------|----------------------------------------------------------|
| Telegram-доставка     | Проверка на VPS | `api.telegram.org` недоступен с локальной машины         |
| Проверка модели DeepSeek | Сейчас        | Требуется живой ключ + `--check-llm`                     |
| URL-нормализация      | Отложена       | UTM-параметры, trailing slash — не убираются             |
| Этап 3 (RSSHub, USGS, Steam) | Не начат | ТГ-каналы через RSSHub; землетрясения через USGS GeoJSON |
| Этап 5 (Docker / VPS) | ✅ Готово        | `Dockerfile`, `docker-compose.yml`, `DEPLOY.md`          |
| at-least-once (send ↔ set_state) | Сейчас | Окно между send и set_state('delivered'); редкий дубль   |
| Хвост retry-очереди   | Сейчас         | Голодание при stuck > max_per_cycle                      |

## Требования

- Python 3.8+

## Деплой на VPS (Docker)

См. [DEPLOY.md](./DEPLOY.md) — полная инструкция: подготовка сервера,
`.env`, привязка chat_id, прокси для Telegram/DeepSeek, бэкап БД,
TZ и расписание дайджеста, обновление.