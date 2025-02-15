FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG USER_ID=ubuntu

RUN useradd -ms /bin/bash $USER_ID
RUN echo $USER_ID:$USER_ID | chpasswd
RUN echo "$USER_ID ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    vim \
    python3-pip \
    libyaml-cpp-dev \
    libpython3.10 \
    libx11-6 \
    libsm6 \
    libxt6 \
    libglib2.0-0 \
    python3-sphinx \
    python3-numpydoc \
    python3-sphinx-rtd-theme \
    python3-pyqt5 \
    python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

# libeigen3-dev install does not work with apt
RUN mkdir -p /root/eigen && \
    cd /root/eigen && \
    wget -O Eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && \
    unzip Eigen.zip && \
    cp -r eigen-3.4.0/Eigen /usr/local/include && \
    cd /root && \
    rm -rf eigen

USER $USER_ID
WORKDIR /home/$USER_ID

RUN pip3 install -U pip
RUN pip3 install PyQt5

RUN git clone https://github.com/Wojcik98/double_pendulum.git double_pendulum

WORKDIR /home/$USER_ID/double_pendulum

RUN make install
RUN make pythonfull
# RUN make doc
# RUN make tests
