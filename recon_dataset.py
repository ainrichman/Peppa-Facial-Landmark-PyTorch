import json
import os
import cv2
import numpy as np
from tqdm import tqdm


def do(filename):
    gogo = json.load(open(filename))
    result = []
    for data in tqdm(gogo):
        image_path = data['image_path'].replace("\\", "/")
        landmarks = np.array(data['keypoints'])
        bbox = np.array(data['bbox'])
        img = cv2.imread(image_path)
        img_h, img_w = img.shape[0], img.shape[1]
        center = np.array([bbox[2] + bbox[0], bbox[1] + bbox[3]]) / 2
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ex_w, ex_h = np.array([w, h]) * [0.4, 0.6] / 2
        x1 = center[0] - w / 2 - ex_w * 2
        y1 = center[1] - h / 2 - ex_h * 2
        x2 = center[0] + w / 2 + ex_w * 2
        y2 = center[1] + h / 2 + ex_h * 2
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        new_bbox = np.array([x1, y1, x2, y2]).astype(np.int)
        landmarks = landmarks - new_bbox[0:2]
        bbox[0:2] = bbox[0:2] - new_bbox[0:2]
        bbox[2:4] = bbox[2:4] - new_bbox[0:2]
        data['bbox'] = bbox.tolist()
        data['keypoints'] = landmarks.tolist()
        img = img[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]
        output_path = os.path.join("/".join(image_path.split("/")[0:-4]), image_path.split("/")[-4] + "_Output",
                                   "/".join(image_path.split("/")[-3:]))
        data['image_path'] = output_path
        output_dir = "/".join(output_path.split("/")[0:-1])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(output_path, img)
        result.append(data)
    json.dump(result, open(filename, "w"))


do("train.json")
do("val.json")
