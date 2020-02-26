import os
import shutil
import numpy as np
import cv2
from scipy.io import loadmat
import shutil


def flip_points(landmarks):
    result = landmarks.copy()
    result[0:8] = landmarks[9:17][::-1]
    result[9:17] = landmarks[0:8][::-1]
    result[17:22] = landmarks[22:27][::-1]
    result[22:27] = landmarks[17:22][::-1]
    result[36] = landmarks[45]
    result[45] = landmarks[36]
    result[37] = landmarks[44]
    result[44] = landmarks[37]
    result[38] = landmarks[43]
    result[43] = landmarks[38]
    result[39] = landmarks[42]
    result[42] = landmarks[39]
    result[40] = landmarks[47]
    result[47] = landmarks[40]
    result[41] = landmarks[46]
    result[46] = landmarks[41]
    result[31:33] = landmarks[34:36][::-1]
    result[34:36] = landmarks[31:33][::-1]
    result[50] = landmarks[52]
    result[52] = landmarks[50]
    result[49] = landmarks[53]
    result[53] = landmarks[49]
    result[48] = landmarks[54]
    result[54] = landmarks[48]
    result[59] = landmarks[55]
    result[59] = landmarks[55]
    result[50] = landmarks[52]
    result[52] = landmarks[50]
    result[58] = landmarks[56]
    result[56] = landmarks[58]
    result[60] = landmarks[64]
    result[64] = landmarks[60]
    result[61] = landmarks[63]
    result[63] = landmarks[61]
    result[67] = landmarks[65]
    result[65] = landmarks[67]
    return result.astype(np.int)


subsets = ["AFW", "AFW_Flip", "HELEN", "HELEN_Flip", "IBUG", "IBUG_Flip", "LFPW", "LFPW_Flip"]

base_path = "H:\\datasets\\300W_LP"
output_base = "H:\\datasets\\300W_LP_Out"
for subset in subsets:
    subset_path = os.path.join(base_path, subset)
    img_files = filter(lambda x: ".jpg" in x, os.listdir(subset_path))
    output_subset = os.path.join(output_base, subset)
    if not os.path.exists(output_subset):
        os.makedirs(output_subset)
    for img_file in img_files:
        mat_file = img_file.replace(".jpg", ".mat")
        output_mat_path = os.path.join(output_subset, img_file.replace(".jpg", ".pts"))
        out = open(output_mat_path, "w")
        out.write("version: 1\nn_points: 68\n{\n")
        mat = loadmat(
            os.path.join(base_path, "landmarks", subset.replace("_Flip", ""), img_file.replace(".jpg", "_pts.mat")))[
            'pts_2d']
        if "Flip" in subset:
            mat[:, 0] = 450 - mat[:, 0]
            mat = flip_points(mat)
        for point in mat:
            out.write(str(point[0]))
            out.write(" ")
            out.write(str(point[1]))
            out.write("\n")
        out.write("}\n")
        img_src = os.path.join(subset_path, img_file)
        img_dst = os.path.join(output_subset, img_file)
        shutil.copy(img_src, img_dst)
        # img = cv2.imread(img_dst)
        # for point in mat:
        #     img = cv2.circle(img,tuple(point),2,(255,0,0),-1,1)
        # cv2.imshow("",img)
        # cv2.waitKey()
