FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
#FROM ubuntu:focal

#Run the frontend first so it doesn't throw an error later
RUN apt-get update \
 && export TZ="America/New_York" \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y locales \
 && ln -fs "/usr/share/zoneinfo/$TZ" /etc/localtime \
 && dpkg-reconfigure --frontend noninteractive tzdata \
 && apt-get clean

# General dependencies for development
RUN apt-get update \
 && apt-get install -y \
        build-essential \
        cmake \
        cppcheck \
        gdb \
        git \
        libeigen3-dev \
        g++ \
        libbluetooth-dev \
        libcwiid-dev \
        libgoogle-glog-dev \
        libspnav-dev \
        libusb-dev \
        lsb-release \
        mercurial \
        python3-dbg \
        python3-empy \
        python3-pip \
        python3-venv \
        software-properties-common \
        sudo \
        wget \
	    curl \
        cmake-curses-gui \
        geany \
        tmux \
        dbus-x11 \
        iputils-ping \
        default-jre \
        iproute2 \
 && apt-get clean

# make a user
ARG user_id
ARG USER ros-user
RUN useradd -U --uid ${user_id} -ms /bin/bash $USER \
 && echo "$USER:$USER" | chpasswd \
 && adduser $USER sudo \
 && echo "$USER ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USER

USER $USER

# Install ROS Noetic
RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \ && sudo /bin/sh -c 'curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -' \
 && sudo apt-get update \
 && sudo apt-get install -y \
    python3-catkin-tools \
    python3-rosdep \
    python3-rosinstall \
    ros-noetic-desktop-full

RUN sudo rosdep init \
 && sudo apt-get clean

RUN rosdep update


# create folders and workspace
WORKDIR /home/$USER
RUN mkdir data

COPY ./LLaVA-NeXT/ ./LLaVA-NeXT/
COPY ./llava-onevision-qwen2-7b-ov-chat/ ./llava-onevision-qwen2-7b-ov-chat/

# install Tenzis dependencies
RUN sudo apt-get install ros-noetic-cv-bridge
RUN pip install hydra-core --upgrade


# install yolov7 dependencies
RUN pip3 install \
 matplotlib \
 numpy \
 opencv-python \
 Pillow \
 PyYAML \
 requests \
 scipy \
 torch \
 torchvision \
 tqdm \
 portalocker['redis']

RUN pip3 install pandas
RUN pip3 install seaborn

RUN pip3 install bitsandbytes transformers==4.37.2 pydantic accelerate==0.21.0 flash-attn==2.6.3

RUN /bin/sh -c 'echo ". /opt/ros/noetic/setup.bash" >> ~/.bashrc'
RUN sudo apt-get install -y ros-noetic-vision-msgs

RUN mkdir -p ws/src
COPY ./ws/src/gone ./ws/src/gone
COPY ./ws/src/tdd2 ./ws/src/tdd2
COPY ./ws/src/dtc_inference ./ws/src/dtc_inference

RUN cd ws/src/ \
 && git clone https://github.com/KumarRobotics/MOCHA

RUN pip3 install --upgrade pip
RUN pip3 install flash-attn --no-build-isolation
RUN cd LLaVA-NeXT && pip3 install .

RUN cd ~/ws \
 && catkin config --extend /opt/ros/noetic --skiplist laser_gazebo_plugins_organized \
 && catkin build --no-status

COPY ./install/download.py download.py
#RUN python3 download.py
RUN pip3 install peft zmq

RUN sudo apt install vim -y

RUN sudo chown $USER:$USER ~/.bashrc \
 && /bin/sh -c 'echo sudo chown -R $USER:$USER /home/dtc/.cache/huggingface/ >> ~/.bashrc' \
 && /bin/sh -c 'echo ". /opt/ros/noetic/setup.bash" >> ~/.bashrc' \
 && /bin/sh -c 'echo "source ~/ws/devel/setup.bash" >> ~/.bashrc' \
 && echo 'export PS1="\[$(tput setaf 2; tput bold)\]\u\[$(tput setaf 7)\]@\[$(tput setaf 3)\]\h\[$(tput setaf 7)\]:\[$(tput setaf 4)\]\W\[$(tput setaf 7)\]$ \[$(tput sgr0)\]"' >> ~/.bashrc
