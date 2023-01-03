FROM python:3.8.16

WORKDIR /app

COPY API/ .

RUN pip3 install --no-cache-dir -r requirements.txt


CMD ["python3", "API.py"]