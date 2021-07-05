FROM python:3.9-alpine
ENV PYTHONUNBUFFERED 1

# Copy and install dependencies
COPY requirements.txt /requirements.txt

RUN apk add --no-cache --virtual .build-deps \
    ca-certificates gcc postgresql-dev linux-headers musl-dev \
    libffi-dev jpeg-dev zlib-dev \
    && pip install -r requirements.txt

# Creating working directory
RUN mkdir /app
WORKDIR /app

# Expose ports
EXPOSE 8000
