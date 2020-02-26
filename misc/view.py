import cv2
import json
import numpy as np

labels = json.load(open("../val.json"))
for label in labels:
    img= cv2.imread(label['image_path'])
    landmarks = np.array(label['keypoints']).reshape((68,2)).astype(np.int)
    for point in landmarks:
        img = cv2.circle(img,tuple(point),2,(255,0,0),-1,1)
    cv2.imshow("",img)
    cv2.waitKey()