FROM python:3.7.6-buster

ENV PYTHONUNBUFFERED 1


RUN useradd -ms /bin/bash admin

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

RUN python -m nltk.downloader punkt

USER admin

EXPOSE 4000

ENTRYPOINT ['python']

CMD ['app.py']

