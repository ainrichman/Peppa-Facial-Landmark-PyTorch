import cv2
from face_onnx.detector import Detector as FaceDetector

'''
Three detector options: 
1. Original PyTorch inference detector 
2. MNN Python inference detector (experimental) 
3. ONNX inference detector based on onnxruntime

MNN detector is only tested on Windows 10 and Centos7.
'''

# from detector import Detector
from mnn_detector import Detector
# from onnx_detector import Detector
import numpy as np

face_detector = FaceDetector()
lmk_detector = Detector()
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (frame.shape[1], frame.shape[0]))
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    bboxes, _ = face_detector.detect(frame)
    if len(bboxes) != 0:
        bbox = bboxes[0]
        bbox = bbox.astype(np.int)
        lmks, PRY_3d = lmk_detector.detect(frame, bbox)
        lmks = lmks.astype(np.int)
        frame = cv2.rectangle(frame, tuple(bbox[0:2]), tuple(bbox[2:4]), (0, 0, 255), 1, 1)
        for point in lmks:
            frame = cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1, 1)
        frame = cv2.putText(frame, "Pitch: {:.4f}".format(PRY_3d[0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 1, 1)
        frame = cv2.putText(frame, "Yaw: {:.4f}".format(PRY_3d[1]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 1, 1)
        frame = cv2.putText(frame, "Roll: {:.4f}".format(PRY_3d[2]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 1, 1)
    cv2.imshow("Peppa Landmark Detection", frame)
    if cv2.waitKey(27) == ord("q"):
        break
    out.write(frame)

out.release()
