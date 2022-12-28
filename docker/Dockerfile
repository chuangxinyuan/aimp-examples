ARG ROOT_CONTAINER=continuumio/miniconda3:4.12.0

ARG BASE_CONTAINER=$ROOT_CONTAINER
FROM $BASE_CONTAINER

LABEL maintainer="Niuhongxing, niuhx@shaiic.com"

USER root
SHELL ["/bin/bash", "-c"]

RUN apt-get -q update && \
    apt-get install -yq --no-install-recommends \
    tini \
    jq \
    curl \
    vim \
    zip \
    unzip \
    wget \
    # install nslookup and dig
    dnsutils \
    sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install requests && \
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers==4.22.0 && \
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.23.3 && \
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple aimp-sdk==0.0.4 && \
    ####################
    # To support opencv, MUST do exactly the same including versions and sequence
    # MUST install this version to use cv2
    # MUST install in the following sequence, no change
    pip3 install opencv-contrib-python==4.5.5.62 && \
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless==4.6.0.66
    ####################

