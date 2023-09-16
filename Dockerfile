# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# The image below is a pinned version of nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 (from Jan 2018)
# If updating the base image, be sure to test on GPU since it has broken in the past.
# UBUNTU 20.04 - cuda 11.4
FROM nvidia/cuda@sha256:e0bd529971d4bddcde35e7658a8089acac6d43fb8d815809cc3913ab0c3977f0

# UBUNTU 22.04 - cuda 11.8
# FROM nvidia/cuda@sha256:0c4830108130fe92fa3e9ba8a9a813bf1264e2cddd1902b2c30750aada5ede38

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 


RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.8-dev python3.8 python3-pip
RUN virtualenv --python=python3.8 env

RUN ln -s /env/bin/python3.8 /usr/bin/python

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN pip install --no-cache-dir gym==0.26.2
RUN pip install --no-cache-dir mujoco==2.3.0
RUN pip install --no-cache-dir h5py==3.7.0
RUN pip install --no-cache-dir imageio==2.22.4
RUN pip install --no-cache-dir opencv-python==4.6.0.66
RUN pip install --no-cache-dir pathos==0.3.0
RUN pip install --no-cache-dir tqdm==4.64.1
RUN pip install --no-cache-dir mujoco-python-viewer==0.1.2
RUN pip install --no-cache-dir pyyaml==6.0
RUN pip install --no-cache-dir PyOpenGL==3.1.7
RUN pip install --no-cache-dir PyOpenGL_accelerate==3.1.7
RUN pip install --no-cache-dir pip install matplotlib==3.6.2
RUN pip install --no-cache-dir pip install scipy==1.9.3
RUN pip install --no-cache-dir wandb
RUN pip install --no-cache-dir imageio
RUN pip install --no-cache-dir moviepy


RUN mkdir /projects; 
RUN git clone https://github.com/glfw/glfw.git /projects/glfw

RUN apt-get update
RUN apt-get install -y g++ curl python-opengl xvfb xorg-dev cmake libzmq3-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev

RUN cd /projects/glfw; \
    cmake -DBUILD_SHARED_LIBS=ON .; \
    make;

ENV PYGLFW_LIBRARY=/projects/glfw/src/libglfw.so
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

