#!/bin/bash
docker build -t plantfinder:latest .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 590183911272.dkr.ecr.us-east-1.amazonaws.com
docker tag plantfinder:latest 590183911272.dkr.ecr.us-east-1.amazonaws.com/plantfinder-0:latest
docker push 590183911272.dkr.ecr.us-east-1.amazonaws.com/plantfinder-0:latest
echo "Plantfinder docker image pushed to ec2 repository as latest version"