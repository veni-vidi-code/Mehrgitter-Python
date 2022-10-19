FROM python:latest
WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD gunicorn -w 4 --bind 0.0.0.0:8000 --forwarded-allow-ips=* wsgi:application
