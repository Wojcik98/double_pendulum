ARG BASE_IMAGE=double_pendulum_base
FROM ${BASE_IMAGE}

ARG USER_ID=ubuntu

USER root
RUN rm /etc/sudoers
RUN apt-get update && apt-get install -y \
    sudo \
    less \
    curl && \
    rm -rf /var/lib/apt/lists/*
RUN echo "$USER_ID ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# TODO: handle sudo install in base image

USER $USER_ID
# Install `just` tool to /usr/bin and generate bash completions
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin && \
    ~/.local/bin/just --completions bash > ~/.just-completions.bash && \
    echo ". ~/.just-completions.bash" >> ~/.bashrc

RUN git clone https://github.com/leggedrobotics/rsl_rl ~/rsl_rl && \
    cd ~/rsl_rl && \
    pip3 install -e . --config-settings editable_mode=strict

RUN pip3 install mpc

# SHELL ["/bin/bash", "-c"]
# COPY entrypoint.sh entrypoint.sh
# RUN chmod +x entrypoint.sh

ENTRYPOINT ["/home/$USER_ID/entrypoint.sh"]
CMD ["/bin/bash"]
