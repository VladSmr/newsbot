# ── Runtime image ─────────────────────────────────────────────────────
FROM python:3.12-slim

# Устанавливаем tzdata ДО ENV TZ — python:3.12-slim не содержит tzdata.
# Без этого TZ=Europe/Moscow игнорируется и часы уезжают на UTC.
# ENV DEBIAN_FRONTEND=noninteractive: нет интерактивного диалога tzselect.
# cp /usr/share/zoneinfo + echo в /etc/timezone — гарантирует tz вне зависимости
# от переменной TZ (belt-and-suspenders).
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && cp /usr/share/zoneinfo/Europe/Moscow /etc/localtime \
    && echo "Europe/Moscow" > /etc/timezone

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