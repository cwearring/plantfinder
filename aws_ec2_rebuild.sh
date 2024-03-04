#!/bin/bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 590183911272.dkr.ecr.us-east-1.amazonaws.com
docker pull 590183911272.dkr.ecr.us-east-1.amazonaws.com/plantfinder-0:latest
docker kill plantfinder 
docker rm plantfinder 
docker run -d --name plantfinder -p 80:8080 -v $(pwd)/logfiles:/app/logfiles 590183911272.dkr.ecr.us-east-1.amazonaws.com/plantfinder-0
# docker run  -d --name plantfinder -p 80:8080-v $(pwd)/logfiles:/app/logfiles  $(docker images | grep latest | awk '{print $3}')
echo "Plantfinder is running on port 80"
# docker exec -it plantfinder /bin/bash
