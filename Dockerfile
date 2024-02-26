# syntax=docker/dockerfile:1

# docker run --name mypostgres --network mynetwork -e POSTGRES_PASSWORD=cwearring -p 5432:5432 -d postgres
# nc -zv localhost 5432
# docker build -t plantfinder .
# docker run --name plantfinder --network mynetwork -e FLASK_ENV=justatest -p 8080:8080 -d -v $(pwd)/logfiles:/app/logfiles -v ($pwd)/OrderForms:/app/OrderForms plantfinder 
# docker exec -it plantfinder /bin/bash
# docker logs plantfinder

#FROM arm64/python:3.10-slim AS base
FROM python:3.10-slim AS base
RUN pip install --upgrade pip
#RUN apk add --no-cache g++ curl cmake protobuf -- for 3.10-alpine 
RUN apt-get update && apt-get -y install g++ curl cmake protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# PYTHONDONTWRITEBYTECODE - Prevents Python from writing pyc files.
# PYTHONUNBUFFERED - Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
# CONFIG_LEVEL - set app to use postgres sql on docker private network 
# CONFIG_LEVEL = [dev, dockr_local, aws_dev1, prod]
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CONFIG_LEVEL="aws_dev1" 

# change working directory
WORKDIR /app
# install the python environment 
COPY requirements.txt /app
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy local code to the container image.
COPY . /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on.
EXPOSE 8080

CMD gunicorn 'app:myapp' --bind=0.0.0.0:8080 \
    --workers 1 \
    --timeout 600

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8080/health || exit 1
