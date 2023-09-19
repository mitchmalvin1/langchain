FROM python:3.10.0

WORKDIR /app
COPY requirements.txt .
COPY src/* .


RUN pip install -r requirements.txt

EXPOSE 3031

ENTRYPOINT ["uwsgi", "--socket", ":3031", "--master", "-p", "1", "-w", "app:app"]