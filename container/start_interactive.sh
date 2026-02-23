#!/bin/bash

#allow docker to connect to xhost
xhost +local:docker

DIR=$(pwd)/../

docker run --gpus all --name boomer_localization -it --rm  \
   	--env DISPLAY=unix$DISPLAY \
   	--ipc=host \
   	-e NVIDIA_DRIVER_CAPABILITIES=all \
   	-e NVIDIA-VISIBLE_DEVICES=all \
   	-e QT_X11_NO_MITSHM=1 \
   	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--net=host\
    --volume="$DIR:/boomer_localization" \
    --gpus all\
    --privileged\
   	-e NB_UID=1000 -e NB_GID=1000 -e VNC_SERVER_PASSWORD=password \
   	-p 6006:6006 -p 5900:5900 -p 8888:8888 \
	--runtime=nvidia\
    boomer_localization bash -c "cd boomer_localization && bash"

