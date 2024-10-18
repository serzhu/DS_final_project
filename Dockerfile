FROM python:3.10-slim-bullseye

WORKDIR /app

COPY . .

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get clean autoclean && apt-get autoremove --yes && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app/app.py"]
