FROM python:latest

# Send Python output to the terminal
ENV PYTHONUNBUFFERED=1

WORKDIR /fcstools

COPY . /fcstools

RUN pip install --upgrade pip && \
    pip install --use-pep517 -e .


