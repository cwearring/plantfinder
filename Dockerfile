# syntax=docker/dockerfile:1

# FOR RUNNING POSTGRES IN LOCAL DOCKER  
# docker run --name mypostgres --network mynetwork -e POSTGRES_PASSWORD=cwearring -p 5432:5432 -d postgres
# nc -zv localhost 5432
# RUN PLANTFINDER DOCKER IMAGE LOCALLY 
# docker network create mynetwork 
# docker run --name plantfinder --network mynetwork -p 8080:8080 -d -v $(pwd)/logfiles:/app/logfiles plantfinder 
# nc -zv localhost 8080
# FOR PUSHING DOCKER IMAGE TO AWS EC2 - EC2 uses http port 80 
# docker build -t plantfinder:latest .
# aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 590183911272.dkr.ecr.us-east-1.amazonaws.com
# docker tag plantfinder:latest 590183911272.dkr.ecr.us-east-1.amazonaws.com/plantfinder-0:latest
# docker push 590183911272.dkr.ecr.us-east-1.amazonaws.com/plantfinder-0:latest
# LOGIN TO EC2 CONTAINER 
# ssh -i ./plant0.pem ec2-user@ec2-44-204-77-209.compute-1.amazonaws.com
# sudo yum update -y
# sudo amazon-linux-extras install docker
# sudo service docker start
# sudo usermod -a -G docker ec2-user  
# getent group docker | cut -d: -f4
# aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 590183911272.dkr.ecr.us-east-1.amazonaws.com
# docker pull 590183911272.dkr.ecr.us-east-1.amazonaws.com/plantfinder-0:latest
# docker kill plantfinder 
# docker rm plantfinder 
# docker run --name plantfinder -p 80:8080 -d -v $(pwd)/logfiles:/app/logfiles  $(docker images | grep latest | awk '{print $3}')
# docker logs -f plantfinder
# LOGIN TO THE CONTAINER FROM LOCAL CLIENT 
# docker exec -it plantfinder /bin/bash


#FROM arm64/python:3.10-slim AS base
FROM python:3.10-slim AS base
RUN pip install --upgrade pip
#RUN apk add --no-cache g++ curl cmake protobuf -- for 3.10-alpine 
RUN apt-get update && apt-get -y install g++ curl cmake protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# create local hugging face cache directory 
RUN mkdir -p /cache/huggingface && chmod 777 /cache/huggingface

# PYTHONDONTWRITEBYTECODE - Prevents Python from writing pyc files.
# PYTHONUNBUFFERED - Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
# CONFIG_LEVEL - set app to use postgres sql on docker private network 
# CONFIG_LEVEL = [dev, dockr_local, aws_dev1, prod]
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface HF_HOME=/cache/huggingface
# CONFIG_LEVEL="aws_dev1" 

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
