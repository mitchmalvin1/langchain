FROM python:3.10.0

WORKDIR /app
COPY ./ .

RUN pip install -r requirements.txt

ENTRYPOINT ["flask", "run"]