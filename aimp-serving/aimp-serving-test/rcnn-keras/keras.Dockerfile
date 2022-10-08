FROM python:3.6

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         git \
         curl && \
     rm -rf /var/lib/apt/lists/*

COPY kerasserver kerasserver
COPY models models
RUN git clone --depth 1 https://github.com/matterport/Mask_RCNN.git
RUN curl -L https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 -o models/mask_rcnn_coco.h5

RUN pip install --no-cache-dir -e ./kerasserver
RUN pip install 'h5py<3.0.0'

ENTRYPOINT ["python", "-m", "kerasserver"]