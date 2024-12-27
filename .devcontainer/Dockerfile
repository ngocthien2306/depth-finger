FROM nvidia/cuda:12.6.0-base-ubuntu20.04

USER root

# Set timezone
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install basic tools
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    gpg \
    libgtk2.0-dev \
    locales \
    lsb-release \
    parallel \
    sudo \
    unzip \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install MetavisionSDK
COPY ./metavision.list /etc/apt/sources.list.d/metavision.list

RUN apt-get update && \
    apt-get install -y \
    metavision-hal-bin \
    metavision-hal-prophesee-plugins \
    metavision-sdk-core-bin \
    metavision-sdk-core-python \
    metavision-sdk-core-python3.8 \
    metavision-sdk-python3.8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python and pip packages
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Set environment variables
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
RUN locale-gen en_US.UTF-8

# Configure user
ARG USER="vscode"
ARG USER_UID=1001
ARG USER_GID=${USER_UID}

RUN groupadd --gid ${USER_GID} ${USER} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USER} && \
    echo ${USER} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USER} && \
    chmod 0440 /etc/sudoers.d/${USER} && \
    usermod --shell /bin/bash ${USER} && \
    usermod -aG plugdev ${USER} && \
    usermod -aG root ${USER} && \
    mkdir -p /home/${USER}/.vscode-server/extensions /home/${USER}/.vscode-server-insiders/extensions && \
    chown -R ${USER} /home/${USER}/.vscode-server /home/${USER}/.vscode-server-insiders

USER ${USER}
