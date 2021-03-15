#! /bin/bash
sudo docker run -ti \
	--gpus all \
	--name yuan-predictor \
	-v /home/yuan/docker/predictor/app:/app \
	-v /home/yuan/docker/backend/data:/data \
	-v /home/yuan/docker/backend/output:/output \
	-d yuan-predictor:2.0.0
