#!/bin/bash
docker kill plantfinder 
docker rm plantfinder 
docker build -t plantfinder .
docker run -d --name plantfinder --network mynetwork -p 8080:8080 -v $(pwd)/logfiles:/app/logfiles  plantfinder
echo "Plantfinder is running on port 8080"
docker logs -f plantfinder