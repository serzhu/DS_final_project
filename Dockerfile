FROM python:3.10-slim

WORKDIR /app

COPY . .

COPY requirements.txt .

RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app/app.py"]
