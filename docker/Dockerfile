ARG ROOT_CONTAINER=continuumio/miniconda3:4.12.0

ARG BASE_CONTAINER=$ROOT_CONTAINER
FROM $BASE_CONTAINER

LABEL maintainer="Niuhongxing, niuhx@shaiic.com"

USER root

RUN apt-get -q update && \
    apt-get install -yq --no-install-recommends \
    tini \
    jq \
    curl \
    vim \
    zip \
    unzip \
    wget \
    sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/* 

RUN pip3 install requests && \
    # MUST install this to use cv2
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless && \
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers && \
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy && \
    pip3 install -i  https://pypi.tuna.tsinghua.edu.cn/simple onepanel-sdk==1.0.1 && \
    # MUST install this version to use cv2
    pip3 install opencv-contrib-python==4.5.5.62
