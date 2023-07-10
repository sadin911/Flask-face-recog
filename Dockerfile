FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "index:app", "-b", "0.0.0.0:5000"]
