#!/bin/bash
docker build --build-arg user_id=$(id -u) --build-arg USER=$(whoami) --rm -t ros-docker:$(whoami) .
