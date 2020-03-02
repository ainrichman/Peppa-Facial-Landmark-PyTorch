import torch
import cv2
from models.slim import Slim
import numpy as np
from tracker import Tracker
from utils.headpose import get_head_pose
import time


class Detector:
    def __init__(self, detection_size=(160, 160)):
        self.model = Slim()
        self.model.load_state_dict(torch.load(open("pretrained_weights/slim_160_latest.pth", "rb"), map_location="cpu"))
        self.model.eval()
        self.model.cuda()
        self.tracker = Tracker()
        self.detection_size = detection_size

    def crop_image(self, orig, bbox):
        bbox = bbox.copy()
        image = orig.copy()
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        face_width = (1 + 2 * 0.25) * bbox_width
        face_height = (1 + 2 * 0.25) * bbox_height
        center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
        bbox[0] = max(0, center[0] - face_width // 2)
        bbox[1] = max(0, center[1] - face_height // 2)
        bbox[2] = min(image.shape[1], center[0] + face_width // 2)
        bbox[3] = min(image.shape[0], center[1] + face_height // 2)
        bbox = bbox.astype(np.int)
        crop_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        h, w, _ = crop_image.shape
        crop_image = cv2.resize(crop_image, self.detection_size)
        return crop_image, ([h, w, bbox[1], bbox[0]])

    def detect(self, img, bbox):
        crop_image, detail = self.crop_image(img, bbox)
        crop_image = (crop_image - 127.0) / 127.0
        crop_image = np.array([np.transpose(crop_image, (2, 0, 1))])
        crop_image = torch.tensor(crop_image).float().cuda()
        with torch.no_grad():
            start = time.time()
            raw = self.model(crop_image)[0].cpu().numpy()
            end = time.time()
            print("PyTorch Inference Time: {:.6f}".format(end - start))
            landmark = raw[0:136].reshape((-1, 2))
        landmark[:, 0] = landmark[:, 0] * detail[1] + detail[3]
        landmark[:, 1] = landmark[:, 1] * detail[0] + detail[2]
        landmark = self.tracker.track(img, landmark)
        _, PRY_3d = get_head_pose(landmark, img)
        return landmark, PRY_3d[:, 0]
