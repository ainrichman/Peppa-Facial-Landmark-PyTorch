import onnxruntime as rt
import cv2
import time
import numpy as np
from face_onnx.prior_box import PriorBox
import os

cfg = {
    'name': 'Slim',
    'min_dim': 128,
    'feature_maps': [[20, 20], [10, 10], [5, 5], [3, 3]],
    'aspect_ratios': [[1], [1]],
    'variance': [0.1, 0.2],
    'clip': False,
}


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return dets[keep]


def decode_raw(raw_detections):
    raw_detections = raw_detections[raw_detections[..., 4] > 0.5]
    if len(raw_detections) == 0:
        return np.array([])
    return np.array(nms(raw_detections, 0.3))


def decode(raw, priors, variances):
    boxes = np.concatenate((
        priors[:, 0:2] + raw[:, 0:2] * variances[0] * priors[:, 2:4],
        priors[:, 2:4] * np.exp(raw[:, 2:4] * variances[1]), raw[:, 4:]), 1)
    boxes[:, 0:2] -= boxes[:, 2:4] / 2
    boxes[:, 2:4] += boxes[:, 0:2]
    return boxes


class Detector:
    def __init__(self, detection_size=(160, 160)):
        dirname = os.path.dirname(__file__)
        self.sess = rt.InferenceSession(os.path.join(dirname, "slim_160.onnx"))
        self.input_name = self.sess.get_inputs()[0].name
        self.variance = [0.1, 0.2]
        self.empty = np.zeros((*detection_size, 3))
        self.detection_size = detection_size
        self.priors = PriorBox(cfg, image_size=detection_size).forward()

    def detect(self, orig):
        orig_h, orig_w, _ = orig.shape
        img = cv2.resize(orig, self.detection_size)
        scale = np.array([orig.shape[1], orig.shape[0], orig.shape[1], orig.shape[0]])
        img = np.transpose(img, (2, 0, 1)) / 255
        img = np.array([img]).astype(np.float32)
        raw = self.sess.run(None, {self.input_name: img})[0]
        raw = raw.reshape((-1, 5))
        raw = decode(raw, self.priors, self.variance)
        dets = decode_raw(raw)
        if len(dets) == 0:
            return np.array([]), np.array([])
        bboxes = dets[:, 0:4]
        bboxes = bboxes * scale
        confs = dets[:, 4]
        bboxes = bboxes.astype(np.int)
        return bboxes, confs


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    detector = Detector()
    total = 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        start = time.time()
        faces, confs = detector.detect(frame)
        end = time.time()
        total += (end - start)
        count += 1
        print(total / count)
        for i, face in enumerate(faces):
            frame = cv2.rectangle(frame, tuple(face[0:2]), tuple(face[2:4]), (255, 0, 0), 2, 1)
            frame = cv2.putText(frame, str(confs[i]), tuple(face[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1,
                                1)
        cv2.imshow("", frame)
        cv2.waitKey(27)
