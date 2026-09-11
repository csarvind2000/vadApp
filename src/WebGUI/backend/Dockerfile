FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/
RUN chmod +x /app/entrypoint.sh 2>/dev/null || true

ENV PYTHONUNBUFFERED=1 DJANGO_SETTINGS_MODULE=vadapp.settings
EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate --noinput && \
     python manage.py collectstatic --noinput && \
     gunicorn vadapp.wsgi:application --bind 0.0.0.0:8000 --workers 3 --timeout 300"]
