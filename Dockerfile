FROM python:3.8-slim
WORKDIR /app
COPY . /app

# install curl 

RUN apt-get update
RUN apt-get install build-essential -y
RUN apt-get install curl -y

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
ENV FLASK_APP=app.py
VOLUME /app/Dataset
VOLUME /app/Backup
CMD ["flask", "run", "--host=0.0.0.0"]
