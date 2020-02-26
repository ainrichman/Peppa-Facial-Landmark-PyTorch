from torch.utils.data import Dataset
from utils.visual_augmentation import ColorDistort, pixel_jitter
import numpy as np
import copy
import json
import random
import cv2
from utils.augmentation import Rotate_aug, Affine_aug, Mirror, Padding_aug, Img_dropout
from utils.headpose import get_head_pose
import time
from utils.turbo.TurboJPEG import TurboJPEG

jpeg = TurboJPEG()

symmetry = [(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9), (8, 8),
            (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
            (31, 35), (32, 34),
            (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
            (48, 54), (49, 53), (50, 52), (55, 59), (56, 58), (60, 64), (61, 63), (65, 67)]
base_extend_range = [0.2, 0.3]


class data_info(object):
    def __init__(self, ann_json):
        self.ann_json = ann_json
        self.metas = []
        self.load_anns()

    def load_anns(self):
        with open(self.ann_json, 'r') as f:
            train_json_list = json.load(f)
        self.metas = train_json_list

    def get_all_sample(self):
        random.shuffle(self.metas)
        return self.metas


class Landmark(Dataset):
    def __init__(self, ann_file, input_size=(160, 160), training_flag=True):
        super(Landmark, self).__init__()
        self.counter = 0
        self.time_counter = 0
        self.training_flag = training_flag
        self.raw_data_set_size = None
        self.color_augmentor = ColorDistort()
        self.lst = self.parse_file(ann_file)
        self.input_size = input_size

    def __getitem__(self, item):
        """Data augmentation function."""
        dp = self.lst[item]
        fname = dp['image_path']
        keypoints = dp['keypoints']
        bbox = dp['bbox']
        if keypoints is not None:
            if ".jpg" in fname:
                image = jpeg.imread(fname)
                # image = cv2.imread(fname)
            else:
                image = cv2.imread(fname)
            label = np.array(keypoints, dtype=np.float).reshape((-1, 2))
            bbox = np.array(bbox)
            crop_image, label = self.augmentationCropImage(image, bbox, label, self.training_flag)

            if self.training_flag:
                if random.uniform(0, 1) > 0.5:
                    crop_image, label = Mirror(crop_image, label=label, symmetry=symmetry)
                if random.uniform(0, 1) > 0.0:
                    angle = random.uniform(-45, 45)
                    crop_image, label = Rotate_aug(crop_image, label=label, angle=angle)
                if random.uniform(0, 1) > 0.5:
                    strength = random.uniform(0, 50)
                    crop_image, label = Affine_aug(crop_image, strength=strength, label=label)
                if random.uniform(0, 1) > 0.5:
                    crop_image = self.color_augmentor(crop_image)
                if random.uniform(0, 1) > 0.5:
                    crop_image = pixel_jitter(crop_image, 15)
                if random.uniform(0, 1) > 0.5:
                    crop_image = Img_dropout(crop_image, 0.2)
                if random.uniform(0, 1) > 0.5:
                    crop_image = Padding_aug(crop_image, 0.3)
            reprojectdst, euler_angle = get_head_pose(label, crop_image)
            PRY = euler_angle.reshape([-1]).astype(np.float32) / 90.
            cla_label = np.zeros([4])
            if dp['left_eye_close']:
                cla_label[0] = 1
            if dp['right_eye_close']:
                cla_label[1] = 1
            if dp['mouth_close']:
                cla_label[2] = 1
            if dp['big_mouth_open']:
                cla_label[3] = 1
            crop_image_height, crop_image_width, _ = crop_image.shape
            # for point in label:
            #     crop_image = cv2.circle(crop_image, tuple(point.astype(np.int)), 3, (255, 0, 0), -1, 1)
            # cv2.imshow("", crop_image)
            # cv2.waitKey()

            label = label.astype(np.float32)
            label[:, 0] = label[:, 0] / crop_image_width
            label[:, 1] = label[:, 1] / crop_image_height

            crop_image = crop_image.astype(np.float32)
            label = label.reshape([-1]).astype(np.float32)
            cla_label = cla_label.astype(np.float32)
            label = np.concatenate([label, PRY, cla_label], axis=0)

        crop_image = (crop_image - 127.0) / 127.0
        crop_image = np.transpose(crop_image, (2, 0, 1)).astype(np.float32)
        return crop_image, label

    def __len__(self):
        return len(self.lst)

    def parse_file(self, ann_file):
        ann_info = data_info(ann_file)
        all_samples = ann_info.get_all_sample()
        self.raw_data_set_size = len(all_samples)
        print("Raw Samples: " + str(self.raw_data_set_size))
        if self.training_flag:
            balanced_samples = self.balance(all_samples)
            print("Balanced Samples: " + str(len(balanced_samples)))
            # balanced_samples = all_samples
            pass
        else:
            balanced_samples = all_samples
        return balanced_samples

    def balance(self, anns):
        res_anns = copy.deepcopy(anns)
        lar_count = 0
        for ann in anns:
            if ann['keypoints'] is not None:
                bbox = ann['bbox']
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                if bbox_width < 50 or bbox_height < 50:
                    res_anns.remove(ann)
                left_eye_close = ann['left_eye_close']
                right_eye_close = ann['right_eye_close']
                if left_eye_close or right_eye_close:
                    for i in range(10):
                        res_anns.append(ann)
                if ann['small_eye_distance']:
                    for i in range(20):
                        res_anns.append(ann)
                if ann['small_mouth_open']:
                    for i in range(20):
                        res_anns.append(ann)
                if ann['big_mouth_open']:
                    for i in range(50):
                        res_anns.append(ann)
                if left_eye_close and not right_eye_close:
                    for i in range(40):
                        res_anns.append(ann)
                    lar_count += 1
                if not left_eye_close and right_eye_close:
                    for i in range(40):
                        res_anns.append(ann)
                    lar_count += 1
        return res_anns

    def augmentationCropImage(self, img, bbox, joints=None, is_training=True):
        bbox = np.array(bbox).reshape(4, ).astype(np.float32)
        add = max(img.shape[0], img.shape[1])
        bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=[127., 127., 127.])
        objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])
        bbox += add
        objcenter += add
        joints[:, :2] += add
        gt_width = (bbox[2] - bbox[0])
        gt_height = (bbox[3] - bbox[1])
        crop_width_half = gt_width * (1 + base_extend_range[0] * 2) // 2
        crop_height_half = gt_height * (1 + base_extend_range[1] * 2) // 2
        if is_training:
            min_x = int(objcenter[0] - crop_width_half + \
                        random.uniform(-base_extend_range[0], base_extend_range[0]) * gt_width)
            max_x = int(objcenter[0] + crop_width_half + \
                        random.uniform(-base_extend_range[0], base_extend_range[0]) * gt_width)
            min_y = int(objcenter[1] - crop_height_half + \
                        random.uniform(-base_extend_range[1], base_extend_range[1]) * gt_height)
            max_y = int(objcenter[1] + crop_height_half + \
                        random.uniform(-base_extend_range[1], base_extend_range[1]) * gt_height)
        else:
            min_x = int(objcenter[0] - crop_width_half)
            max_x = int(objcenter[0] + crop_width_half)
            min_y = int(objcenter[1] - crop_height_half)
            max_y = int(objcenter[1] + crop_height_half)
        joints[:, 0] = joints[:, 0] - min_x
        joints[:, 1] = joints[:, 1] - min_y
        img = bimg[min_y:max_y, min_x:max_x, :]
        crop_image_height, crop_image_width, _ = img.shape
        joints[:, 0] = joints[:, 0] / crop_image_width
        joints[:, 1] = joints[:, 1] / crop_image_height
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST,
                          cv2.INTER_LANCZOS4]
        interp_method = random.choice(interp_methods)
        img = cv2.resize(img, (self.input_size[0], self.input_size[1]), interpolation=interp_method)
        joints[:, 0] = joints[:, 0] * self.input_size[0]
        joints[:, 1] = joints[:, 1] * self.input_size[1]
        return img, joints
