import torch
import cv2
import numpy as np
from tracker import Tracker
from utils.headpose import get_head_pose
import time
import onnxruntime as rt


class Detector:
    def __init__(self, detection_size=(160, 160)):
        self.sess = rt.InferenceSession("pretrained_weights/slim_160_latest.onnx")
        self.input_name = self.sess.get_inputs()[0].name
        self.detection_size = detection_size
        self.tracker = Tracker()

    def crop_image(self, orig, bbox):
        bbox = bbox.copy()
        image = orig.copy()
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        add = int(max(bbox_width, bbox_height))
        bimg = cv2.copyMakeBorder(image, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=[127., 127., 127.])
        bbox += add
        face_width = (1 + 2 * 0.05) * bbox_width
        center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
        bbox[0] = center[0] - face_width // 2
        bbox[1] = center[1] - face_width // 2
        bbox[2] = center[0] + face_width // 2
        bbox[3] = center[1] + face_width // 2
        bbox = bbox.astype(np.int)
        crop_image = bimg[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        h, w, _ = crop_image.shape
        crop_image = cv2.resize(crop_image, self.detection_size)
        return crop_image, ([h, w, bbox[1], bbox[0], add])

    def detect(self, img, bbox):
        crop_image, detail = self.crop_image(img, bbox)
        crop_image = (crop_image - 127.0) / 127.0
        crop_image = np.array([np.transpose(crop_image, (2, 0, 1))]).astype(np.float32)
        start = time.time()
        raw = self.sess.run(None, {self.input_name: crop_image})[0][0]
        end = time.time()
        print("Inference Time: {:.6f}".format(end - start))
        landmark = raw[0:136].reshape((-1, 2))
        landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3] - detail[4]
        landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2] - detail[4]
        landmark = self.tracker.track(img, landmark)
        _, PRY_3d = get_head_pose(landmark, img)
        return landmark, PRY_3d[:, 0]
