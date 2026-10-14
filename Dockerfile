# ── Runtime image ─────────────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # TZ нужен для корректного расписания дайджеста (digest.times в локальной tz)
    # переопределяется через docker-compose environment
    TZ=Europe/Moscow

WORKDIR /app

# Устанавливаем зависимости СИСТЕМНО (от root), ДО переключения на newsbot.
# pip ставит в /usr/local/lib/python3.12/site-packages — доступно всем пользователям.
# requirements.txt копируется первым, чтобы слой кэшировался при неизменных зависимостях.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходники и конфиг
COPY newsbot/ ./newsbot/
COPY config.yaml ./

# Создаём рабочий каталог для БД
RUN mkdir -p /app/data && \
    # непривилегированный пользователь для безопасности
    groupadd --gid 1000 newsbot && \
    useradd --uid 1000 --gid 1000 --shell /bin/false --home /app newsbot && \
    chown -R newsbot:newsbot /app

USER newsbot

# ENTRYPOINT — точка входа; CMD переопределяется через compose для --once / --check-*
ENTRYPOINT ["python", "-m", "newsbot"]