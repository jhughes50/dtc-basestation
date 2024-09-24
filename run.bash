#!/bin/bash

# Make sure processes in the container can connect to the x server
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    touch $XAUTH
fi
xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
if [ -n "$xauth_list" ]
then
  echo "$xauth_list" | xauth -f $XAUTH nmerge -
fi
chmod a+r $XAUTH
xhost +
docker run --rm -it --gpus all \
    --network=host \
    -u $UID \
    -e "TERM=xterm-256color" \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "/mnt/dtc/data:/home/`whoami`/data/" \
    -v "/mnt/dtc/perception_models:/mnt/dtc/perception_models" \
    -v "`pwd`/../DTC_internal:/home/`whoami`/ws/src/DTC_internal" \
    -v "/dtc/DTC_2024/challenge/:/home/`whoami`/dtc_challenge" \
    -v "./config/radio_configs.yaml:/home/`whoami`/ws/src/MOCHA/mocha_core/config/radio_configs.yaml" \
    -v "./config/robot_configs.yaml:/home/`whoami`/ws/src/MOCHA/mocha_core/config/robot_configs.yaml" \
    -v "./config/topic_configs.yaml:/home/`whoami`/ws/src/MOCHA/mocha_core/config/topic_configs.yaml" \
    -v "./config/watchstate:/home/`whoami`/ws/src/MOCHA/interface_rajant/scripts/thirdParty/watchstate" \
    -v "/home/mhussing/.cache/huggingface/hub/models--google--siglip-so400m-patch14-384/:/home/dtc/.cache/huggingface/hub/models--google--siglip-so400m-patch14-384/" \
    --name ros-docker-$(whoami) \
    ros-docker:$(whoami) \
    bash
xhost -
