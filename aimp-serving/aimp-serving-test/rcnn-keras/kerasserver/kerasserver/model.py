import kfserving
from typing import Dict
import os
import numpy as np
import sys
from skimage.measure import find_contours, approximate_polygon
import tensorflow as tf
import json
import base64
from PIL import Image
import io

MASK_RCNN_DIR = os.path.abspath(os.environ.get('MASK_RCNN_DIR'))
if MASK_RCNN_DIR:
    sys.path.append(MASK_RCNN_DIR)  # To find local version of the library

from mrcnn import model as modellib
from mrcnn.config import Config
import keras.backend.tensorflow_backend as ktf

class ModelLoader:
    def __init__(self, labels, model_path):
        self.labels = {item['id']: item['name'] for item in labels}
        self.model_path = model_path

        class InferenceConfig(Config):
            NAME = "coco"
            NUM_CLASSES = 1 + 80  # COCO has 80 classes
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        def get_session(gpu_fraction=1.000):
            gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction,
            allow_growth=True)
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        ktf.set_session(get_session())
        # Print config details
        self.config = InferenceConfig()
        self.config.display()

        self.model = modellib.MaskRCNN(mode="inference",
            config=self.config, model_dir=MASK_RCNN_DIR)
        self.model.load_weights(self.model_path, by_name=True)

    def infer(self, image, threshold):
        output = self.model.detect([image], verbose=1)[0]

        results = {"prediction": []}
        prediction = results["prediction"]

        MASK_THRESHOLD = 0.5
        for i in range(len(output["rois"])):
            score = output["scores"][i]
            class_id = output["class_ids"][i]
            mask = output["masks"][:, :, i]
            if score >= threshold:
                mask = mask.astype(np.uint8)
                contours = find_contours(mask, MASK_THRESHOLD)
                # only one contour exist in our case
                contour = contours[0]
                contour = np.flip(contour, axis=1)
                # Approximate the contour and reduce the number of points
                contour = approximate_polygon(contour, tolerance=2.5)
                if len(contour) < 6:
                    continue
                label = self.labels[class_id]

                prediction.append({
                    "confidence": str(score),
                    "label": label,
                    "points": contour.ravel().tolist(),
                    "type": "polygon",
                })

        return results


class KerasModel(kfserving.KFModel):
    def __init__(self, name: str, model_class_name: str, model_dir: str):
        super().__init__(name)
        self.name = name
        self.model_class_name = model_class_name
        self.model_dir = model_dir

        labels = {}
        with open(model_dir + '/labels.json', 'r') as json_file:
            labels = json.load(json_file)

        self.model = ModelLoader(labels, model_dir + '/mask_rcnn_coco.h5')
        self.load()

    def load(self):
        pass

    def predict(self, request: Dict) -> Dict:
        inputs = request["instances"]

        # Input follows the Tensorflow V1 HTTP API for binary values
        # https://www.tensorflow.org/tfx/serving/api_rest#encoding_binary_values
        data = inputs[0]["image"]

        buf = io.BytesIO(base64.b64decode(data["b64"]))
        threshold = float(data.get("threshold", 0.2))
        image = Image.open(buf)

        return self.model.infer(np.array(image), threshold)