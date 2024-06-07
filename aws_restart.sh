#!/bin/bash
# 2024-06-05 ssh -i "./plant0.pem" ec2-user@ec2-18-207-210-29.compute-1.amazonaws.com 
docker kill plantfinder 
docker rm plantfinder 
#docker build -t plantfinder .
#docker run -d --name plantfinder -p 80:8080 -v $(pwd)/logfiles:/app/logfiles plantfinder
docker run -d --name plantfinder -p 80:8080 -v $(pwd)/logfiles:/app/logfiles 590183911272.dkr.ecr.us-east-1.amazonaws.com/plantfinder-0
echo "Plantfinder is running on port 80"