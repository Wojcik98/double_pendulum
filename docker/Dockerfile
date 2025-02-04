FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

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
    python-is-python3

# libeigen3-dev install does not work with apt
RUN wget -O Eigen.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
RUN unzip Eigen.zip
RUN cp -r eigen-3.4.0/Eigen /usr/local/include

#RUN python -m ensurepip --upgrade
RUN pip install -U pip
RUN pip install PyQt5

RUN git clone https://github.com/Wojcik98/double_pendulum.git /double_pendulum

RUN git clone https://github.com/leggedrobotics/rsl_rl /rsl_rl && \
    cd /rsl_rl && \
    pip3 install -e . --config-settings editable_mode=strict

WORKDIR "/double_pendulum"

# RUN git checkout v0.1.0

RUN pip install mpc

RUN make install
RUN make pythonfull
# RUN make doc
# RUN make tests
