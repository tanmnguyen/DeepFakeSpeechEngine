#!/bin/bash

# remove container with the same name
docker stop deepfakespeechengine-app
docker rm deepfakespeechengine-app

# remove the image with the same name
docker rmi deepfakespeechengine-app

docker-compose up --build 

# ------------------------------------------------------------------------------------------------------------
# execute this at the end of the dockerfile 
docker run -t -d --name deepfakespeechengine-app deepfakespeechengine-app

docker cp deepfakespeechengine-app:/app/. .

# stop and remove the container
docker stop deepfakespeechengine-app
docker rm deepfakespeechengine-app