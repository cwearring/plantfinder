#!/bin/bash
docker kill plantfinder 
docker rm plantfinder 
docker build -t plantfinder .
docker run --name plantfinder --network mynetwork -e FLASK_ENV=justatest -p 8080:8080 -d -v $(pwd)/logfiles:/app/logfiles plantfinder