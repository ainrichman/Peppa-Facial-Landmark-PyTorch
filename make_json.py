import os
import random
import numpy as np
import json
import traceback

from tqdm import tqdm

'''
i decide to merge more data from CelebA, the data anns will be complex, so json maybe a better way. 
'''

data_dir = 'H:/datasets/300W_All_Orig'  ########points to your director,300w

train_json = 'train.json'
val_json = 'val.json'
img_size = 160
eye_close_thres = 0.02
mouth_close_thres = 0.02
big_mouth_open_thres = 0.08


def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # if s == "pts":
            #     continue
            newDir = os.path.join(dir, s)
            GetFileList(newDir, fileList)
    return fileList


pic_list = []
GetFileList(data_dir, pic_list)
pic_list = [x for x in pic_list if '.jpg' in x or 'png' in x or 'jpeg' in x]
random.shuffle(pic_list)
ratio = 0.95
train_list = pic_list[:int(ratio * len(pic_list))]
val_list = pic_list[int(ratio * len(pic_list)):]

train_json_list = []
for pic in tqdm(train_list):
    one_image_ann = {}
    one_image_ann['image_path'] = pic
    pts = pic.rsplit('.', 1)[0] + '.pts'
    if os.access(pic, os.F_OK) and os.access(pts, os.F_OK):
        try:
            tmp = []
            with open(pts) as p_f:
                labels = p_f.readlines()[3:-1]
            for _one_p in labels:
                xy = _one_p.rstrip().split(' ')
                tmp.append([float(xy[0]), float(xy[1])])
            one_image_ann['keypoints'] = tmp
            label = np.array(tmp).reshape((-1, 2))
            bbox = [float(np.min(label[:, 0])), float(np.min(label[:, 1])), float(np.max(label[:, 0])),
                    float(np.max(label[:, 1]))]

            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]

            left_eye_close = np.sqrt(
                np.square(label[37, 0] - label[41, 0]) +
                np.square(label[37, 1] - label[41, 1])) / bbox_height < eye_close_thres \
                             or np.sqrt(np.square(label[38, 0] - label[40, 0]) +
                                        np.square(label[38, 1] - label[40, 1])) / bbox_height < eye_close_thres
            right_eye_close = np.sqrt(
                np.square(label[43, 0] - label[47, 0]) +
                np.square(label[43, 1] - label[47, 1])) / bbox_height < eye_close_thres \
                              or np.sqrt(np.square(label[44, 0] - label[46, 0]) +
                                         np.square(
                                             label[44, 1] - label[46, 1])) / bbox_height < eye_close_thres
            small_eye_distance = np.sqrt(np.square(label[36, 0] - label[45, 0]) +
                                         np.square(label[36, 1] - label[45, 1])) / bbox_width < 0.5
            small_mouth_open = np.sqrt(np.square(label[62, 0] - label[66, 0]) +
                                       np.square(label[62, 1] - label[66, 1])) / bbox_height > 0.15
            big_mouth_open = np.sqrt(np.square(label[62, 0] - label[66, 0]) +
                                     np.square(label[62, 1] - label[66, 1])) / img_size > big_mouth_open_thres
            mouth_close = np.sqrt(np.square(label[61, 0] - label[67, 0]) +
                                  np.square(label[61, 1] - label[67, 1])) / img_size < mouth_close_thres \
                          or np.sqrt(np.square(label[62, 0] - label[66, 0]) +
                                     np.square(label[62, 1] - label[66, 1])) / img_size < mouth_close_thres \
                          or np.sqrt(np.square(label[63, 0] - label[65, 0]) +
                                     np.square(label[63, 1] - label[65, 1])) / img_size < mouth_close_thres
            one_image_ann['left_eye_close'] = bool(left_eye_close)
            one_image_ann['right_eye_close'] = bool(right_eye_close)
            one_image_ann['small_eye_distance'] = bool(small_eye_distance)
            one_image_ann['small_mouth_open'] = bool(small_mouth_open)
            one_image_ann['big_mouth_open'] = bool(big_mouth_open)
            one_image_ann['mouth_close'] = bool(mouth_close)

            one_image_ann['bbox'] = bbox
            one_image_ann['attr'] = None
            train_json_list.append(one_image_ann)
        except:
            print(pic)
            traceback.print_exc()

with open(train_json, 'w') as f:
    json.dump(train_json_list, f, indent=2)

val_json_list = []
for pic in tqdm(val_list):
    one_image_ann = {}

    ### image_path
    one_image_ann['image_path'] = pic

    #### keypoints
    pts = pic.rsplit('.', 1)[0] + '.pts'
    if os.access(pic, os.F_OK) and os.access(pts, os.F_OK):
        try:
            tmp = []
            with open(pts) as p_f:
                labels = p_f.readlines()[3:-1]
            for _one_p in labels:
                xy = _one_p.rstrip().split(' ')
                tmp.append([float(xy[0]), float(xy[1])])

            one_image_ann['keypoints'] = tmp

            label = np.array(tmp).reshape((-1, 2))
            bbox = [float(np.min(label[:, 0])), float(np.min(label[:, 1])), float(np.max(label[:, 0])),
                    float(np.max(label[:, 1]))]
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]

            left_eye_close = np.sqrt(
                np.square(label[37, 0] - label[41, 0]) +
                np.square(label[37, 1] - label[41, 1])) / bbox_height < eye_close_thres \
                             or np.sqrt(np.square(label[38, 0] - label[40, 0]) +
                                        np.square(label[38, 1] - label[40, 1])) / bbox_height < eye_close_thres
            right_eye_close = np.sqrt(
                np.square(label[43, 0] - label[47, 0]) +
                np.square(label[43, 1] - label[47, 1])) / bbox_height < eye_close_thres \
                              or np.sqrt(np.square(label[44, 0] - label[46, 0]) +
                                         np.square(
                                             label[44, 1] - label[46, 1])) / bbox_height < eye_close_thres
            small_eye_distance = np.sqrt(np.square(label[36, 0] - label[45, 0]) +
                                         np.square(label[36, 1] - label[45, 1])) / bbox_width < 0.5
            small_mouth_open = np.sqrt(np.square(label[62, 0] - label[66, 0]) +
                                       np.square(label[62, 1] - label[66, 1])) / bbox_height > 0.15
            big_mouth_open = np.sqrt(np.square(label[62, 0] - label[66, 0]) +
                                     np.square(label[62, 1] - label[66, 1])) / img_size > big_mouth_open_thres
            mouth_close = np.sqrt(np.square(label[61, 0] - label[67, 0]) +
                                  np.square(label[61, 1] - label[67, 1])) / img_size < mouth_close_thres \
                          or np.sqrt(np.square(label[62, 0] - label[66, 0]) +
                                     np.square(label[62, 1] - label[66, 1])) / img_size < mouth_close_thres \
                          or np.sqrt(np.square(label[63, 0] - label[65, 0]) +
                                     np.square(label[63, 1] - label[65, 1])) / img_size < mouth_close_thres
            one_image_ann['left_eye_close'] = bool(left_eye_close)
            one_image_ann['right_eye_close'] = bool(right_eye_close)
            one_image_ann['small_eye_distance'] = bool(small_eye_distance)
            one_image_ann['small_mouth_open'] = bool(small_mouth_open)
            one_image_ann['big_mouth_open'] = bool(big_mouth_open)
            one_image_ann['mouth_close'] = bool(mouth_close)
            one_image_ann['bbox'] = bbox
            ### placeholder
            one_image_ann['attr'] = None
            val_json_list.append(one_image_ann)

        except:
            print(pic)

with open(val_json, 'w') as f:
    json.dump(val_json_list, f, indent=2)
