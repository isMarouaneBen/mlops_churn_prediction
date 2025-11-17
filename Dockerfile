FROM python:latest

WORKDIR /app

COPY ./requirements.txt .

COPY ./models ./models

COPY api.py .

RUN pip install -r requirements.txt

CMD ["uvicorn", "api:app", "--host","--host", "0.0.0.0", "--port 80"]