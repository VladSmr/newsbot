# Деплой newsbot на VPS (Docker)

## Требования

- VPS с Docker и docker compose (v2) — **один экземпляр бота на сервере**
- `api.telegram.org` и `api.deepseek.com` должны быть доступны из контейнера
- ОС: Linux (amd64 / arm64). Python 3.12 в образе.

---

## 1. Подготовка сервера

```bash
# Установите Docker + compose, если ещё нет:
#   curl -fsSL https://get.docker.com | sh
#   docker compose version
```

## 2. Копирование проекта на VPS

```bash
# Вариант A: git clone (рекомендуется)
git clone https://github.com/VladSmr/newsbot.git
cd newsbot
# На GitHub основная ветка — main (не master):
git checkout main

# Вариант B: scp / rsync
scp -r ./newsbot user@vps:/opt/newsbot/
ssh user@vps "cd /opt/newsbot && ..."
```

## 3. Файл секретов `.env`

```bash
cd /opt/newsbot

# ВАЖНО: файл .env ДОЛЖЕН существовать до первого docker compose up.
# Иначе docker создаст вместо файла каталог и монтирование сломается.
# Создаём из примера и заполняем:
cp .env.example .env
nano .env   # ← TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DEEPSEEK_API_KEY
chmod 600 .env
# Владелец — uid 1000: контейнер (newsbot, uid 1000) должен дописать TELEGRAM_CHAT_ID
sudo chown 1000:1000 .env
```

> **Безопасность:** `.env` не попадает в образ (`dockerignore` + `env_file`).
> Убедитесь, что `chmod 600 .env` и `chown 1000:1000 .env` на сервере.
>
> ⚠️ **Важно:** если `.env` принадлежит root, `--set-chat-id` упадёт с
> `PermissionError: [Errno 13] Permission denied: '/app/.env'`. Лечится
> `sudo chown 1000:1000 .env`.

---

## 4. Подготовка каталога для БД

Контейнер работает под непривилегированным пользователем `newsbot` (uid 1000).
Bind-mount `./data:/app/data` монтирует каталог хоста внутрь контейнера.

**Если на хосте вы запускаете от root** (uid 0), uid контейнера (1000)
не совпадает с владельцем `data/`, и SQLite не сможет создать файл БД
(ошибка «unable to open database file»).

**Исправление:**

```bash
# Создать каталог и передать его владение uid 1000 (newsbot)
mkdir -p data
sudo chown -R 1000:1000 data
```

Если на хосте вы работаете от того же не-root пользователя — каталог уже
принадлежит вам, ничего делать не нужно.

**Как узнать свой uid:**
```bash
id -u
```

**Диагностика «unable to open database file»:**
```bash
# Зайти в контейнер и проверить
docker compose run --rm --entrypoint sh newsbot -c "ls -la /app/data/ && id"

# Если ls показывает access denied — причина в правах на data/
# Лечение:
sudo chown -R 1000:1000 data
```

> ℹ️ Образ имеет `ENTRYPOINT ["python", "-m", "newsbot"]`, поэтому в
> `docker compose run ... newsbot <args>` команда `python -m newsbot` **не**
> дублируется.
>
> Для **произвольного Python** используйте:
> `docker compose run --rm --entrypoint python newsbot -c "..."`
>
> Для **НЕ-python команд** (`sh`, `env`, `ls` и т.п.) — **обязательно**
> `--entrypoint <cmd>`, иначе команда будет дописана к ENTRYPOINT как аргумент
> Python и вызовет ошибку. Пример:
> `docker compose run --rm --entrypoint sh newsbot -c "ls -la /app/data/"`

## 5. Привязка chat_id

Если `TELEGRAM_CHAT_ID` ещё не известен:

```bash
# 1. Откройте чат с ботом в Telegram и нажмите /start
# 2. На сервере:
docker compose run --rm newsbot --set-chat-id
# В .env на хосте автоматически запишется TELEGRAM_CHAT_ID=<ваш_id>
```

## 6. Проверка связности (перед первым запуском)

```bash
# Проверка доступа к Telegram API
docker compose run --rm newsbot --check-telegram

# Проверка доступа к DeepSeek API (и баланс)
docker compose run --rm newsbot --check-llm
```

> ⚠️ **Не запускайте** эти команды, если демон уже работает
> (`docker compose up -d`). Одновременно **только один** потребитель
> `getUpdates` — будет конфликт 409. Остановите демон перед проверкой:
> `docker compose down`

---

## 7. Запуск

```bash
# Первый запуск (и пересборка если обновили код)
docker compose up -d --build

# Проверка статуса
docker compose ps

# Логи (в реальном времени)
docker compose logs -f

# Логи только бота (без перезапуска)
docker compose logs -f newsbot
```

---

## 8. Обновление

```bash
# Остановить → пересобрать → запустить
docker compose down
docker compose up -d --build
docker compose logs -f
```

Или (менее безопасно для воркера, но быстрее):

```bash
docker compose up -d --build
```

---

## 9. Резервное копирование БД

База `newsbot.db` хранится в `./data/` на хосте (монтируется в `/app/data`).

```bash
# Бэкап
cp data/newsbot.db "data/newsbot.db.backup-$(date +%Y%m%d-%H%M%S)"

# Восстановление
#   docker compose down
#   cp data/newsbot.db.backup-YYYYMMDD-HHMMSS data/newsbot.db
#   docker compose up -d
```

---

## 10. Часовой пояс и расписание дайджеста

`digest.times` в `config.yaml` работает в **локальном времени контейнера**,
которое задаётся переменной `TZ` (по умолчанию `Europe/Moscow`).

**Изменить часовой пояс:**

```bash
# в docker-compose.yml:
environment:
  TZ: Europe/Moscow   # ← поменяйте на свой, например Europe/Kaliningrad
```

Или в `.env`:

```bash
echo "TZ=Europe/Kaliningrad" >> .env
```

Список допустимых значений TZ: `ls /usr/share/zoneinfo/` или
[TZ database](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)
(колонка **TZ**).

> Если `TZ` не совпадает с часовым поясом `digest.times` в `config.yaml`,
> дайджест будет отправляться в неожиданное время. Убедитесь, что оба
> значения согласованы.

---

## 11. Прокси для Telegram и DeepSeek

### Зачем нужен прокси

`api.telegram.org` может быть недоступен напрямую из РФ на некоторых VPS
(или наоборот — нужен прокси для api.deepseek.com). В этом случае настройте
HTTP(S)-прокси.

### Как подключить

В `docker-compose.yml` раскомментируйте блок `environment` с прокси:

```yaml
environment:
  TZ: Europe/Moscow
  # ── ПРОКСИ ────────────────────────────────────────────────────────
  HTTPS_PROXY: "http://proxy.example.com:8080"
  HTTP_PROXY:  "http://proxy.example.com:8080"
  NO_PROXY:    "localhost,127.0.0.1"
```

Если прокси с авторизацией:

```yaml
HTTPS_PROXY: "http://user:password@proxy.example.com"   # например, http://user:pass@proxy.example.com:8080
```

### Как это работает

`httpx.Client` создаётся с `trust_env=True` во **всех** HTTP-клиентах бота:
`collector.py` (RSS-фиды), `telegram_io.py` (long-poll getUpdates),
`delivery/telegram.py` (отправка сообщений через Bot API),
`llm/client.py` (DeepSeek). Это означает:

- `HTTPS_PROXY` — используется для HTTPS-соединений (api.telegram.org, api.deepseek.com)
- `HTTP_PROXY` — для HTTP (feedparser-фиды)
- `NO_PROXY` — хосты, которые идут напрямую (обычно localhost)

> Если прокси указан в `.env` или напрямую в `docker-compose.yml` —
> **не перезапускайте демон и не запускайте** `--check-telegram` одновременно
> (конфликт getUpdates). Сначала `docker compose down`, потом проверка.

### Диагностика

```bash
# Проверить доступность Telegram через прокси
docker compose run --rm newsbot --check-telegram

# Проверить доступность DeepSeek через прокси
docker compose run --rm newsbot --check-llm

# С боевым прокси в environment — запустить в foreground для отладки
docker compose run --rm newsbot --once
```

Если проблема сохраняется — убедитесь, что прокси-сервер действительно
доступен с VPS:
```bash
curl --proxy http://proxy.example.com:8080 https://api.telegram.org/getMe
```

---

## 12. Одновременно только один getUpdates

`TelegramPoller` использует long-polling `getUpdates`. Telegram позволяет только
**один** активный потребитель на токен бота.

> ⚠️ **НЕ делайте так:**
> - Не запускайте `--check-telegram` / `--set-chat-id` при работающем демоне.
> - Не запускайте `--once` одновременно с активным демоном.
> - Не запускайте два контейнера с одним и тем же токеном.

Если видите ошибку `Conflict: bot can only be invoked by its owner` или
429 с `retry_after` — значит, есть другой потребитель getUpdates.
Остановите всё: `docker compose down`, проверьте `docker ps`.

---

## 13. Полезные команды

```bash
# Перезапуск демона
docker compose restart newsbot

# Остановка
docker compose down

# Остановить и удалить volumes (данные потеряются!)
docker compose down -v

# Зайти в контейнер для отладки
docker compose run --rm --entrypoint sh newsbot

# Проверить конфиг без запуска
docker compose run --rm --entrypoint python newsbot -c "from newsbot.config import load_config; print(load_config())"

# Проверить, какие переменные окружения видит контейнер
docker compose run --rm --entrypoint env newsbot | grep -E "TELEGRAM|DEEPSEEK|HTTP_PROXY"
```

## 14. Команды бота (в чате Telegram)

```
/stop     — приостановить обработку и доставку
           (сбор RSS продолжается, новости накопятся)
/continue — возобновить после /stop
/digest   — запустить дайджест отложенных новостей
/help     — справка
```

**Пауза (`/stop`):** RSS-сбор работает, items сохраняются как `state='new'`.
При `/continue` накопленные новости обработаются в ближайшем цикле
(в рамках обычного бюджета `max_per_cycle`).
Флаг хранится в БД — переживает рестарт контейнера.