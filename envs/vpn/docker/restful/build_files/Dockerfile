FROM python:3.9-alpine
ENV PYTHONUNBUFFERED 1

# Copy and install dependencies
COPY requirements.txt /requirements.txt

RUN apk add --no-cache --virtual .build-deps \
    ca-certificates gcc postgresql-dev linux-headers musl-dev \
    libffi-dev jpeg-dev zlib-dev bash \
    && pip install -r requirements.txt

# Creating working directory
RUN mkdir /app
WORKDIR /app

COPY ./entrypoint.sh /

# Expose ports
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
