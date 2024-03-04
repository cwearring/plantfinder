#!/bin/bash
docker kill plantfinder 
docker rm plantfinder 
#docker build -t plantfinder .
#docker run -d --name plantfinder -p 80:8080 -v $(pwd)/logfiles:/app/logfiles plantfinder
docker run -d --name plantfinder -p 80:8080 -v $(pwd)/logfiles:/app/logfiles 590183911272.dkr.ecr.us-east-1.amazonaws.com/plantfinder-0
echo "Plantfinder is running on port 80"