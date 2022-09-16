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
    # install nslookup and dig
    dnsutils \
    sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install requests && \
    # MUST install this to use cv2
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless==4.6.0.66 && \
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers==4.22.0 && \
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.23.3 && \
    pip3 install -i  https://pypi.tuna.tsinghua.edu.cn/simple onepanel-sdk==1.0.1 && \
    # MUST install this version to use cv2
    pip3 install opencv-contrib-python==4.5.5.62

RUN conda config --add channels conda-forge && \
    conda update --all --yes && \
    conda create -c conda-forge -n go1.15 go=1.15 -y

RUN source activate go1.15 && \
    go env -w GO111MODULE=on && \
    go env -w GOPROXY=https://goproxy.cn,direct
