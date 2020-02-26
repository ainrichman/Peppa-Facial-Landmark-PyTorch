import cv2
import numpy as np
import math

lk_params = dict(winSize=(40, 40), maxLevel=8, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.1))


def dist(a, b):
    return np.sum(np.power(a - b, 2), 1)


class LKTracker:
    def __init__(self):
        self.prev_frame = None
        self.prev_points = None

    def delta_fn(self, prev_points, new_detected, lk_tracked):
        result = np.zeros(new_detected.shape)
        dist_detect = dist(new_detected, prev_points)
        dist_lk =  dist(new_detected,lk_tracked)
        eye_indices = list(set(range(36, 48)))
        rest_indices = np.array(list(set(range(68)) - set(range(36, 48))))
        eye_indices = np.array(eye_indices)

        dist_detect_eyes = dist_detect[eye_indices]
        dist_detect_rest = dist_detect[rest_indices]

        detect_eyes = new_detected[eye_indices]
        lk_eyes = lk_tracked[eye_indices]

        detect_rest = new_detected[rest_indices]
        lk_rest = lk_tracked[rest_indices]
        temp = result[eye_indices]
        thres = 1
        weight1 = 0.80  # Trust lk when less than thres
        weight2 = 0.85  # Trust Detector when more than thres
        temp[dist_detect_eyes >= thres] = detect_eyes[dist_detect_eyes >= thres] * weight2 + lk_eyes[
            dist_detect_eyes >= thres] * (1 - weight2)
        temp[dist_detect_eyes < thres] = lk_eyes[dist_detect_eyes < thres] * weight1 + detect_eyes[
            dist_detect_eyes <= thres] * (1 - weight1)
        result[eye_indices] = temp

        thres = 10
        temp = result[rest_indices]
        temp[dist_detect_rest < thres] = lk_rest[dist_detect_rest < thres] * weight1 + detect_rest[
            dist_detect_rest < thres] * (1 - weight1)
        temp[dist_detect_rest >= thres] = detect_rest[dist_detect_rest >= thres] * weight2 + lk_rest[
            dist_detect_rest >= thres] * (1 - weight2)
        result[rest_indices] = temp
        return np.array(result)

    def lk_track(self, next_frame, new_detected_points):
        if self.prev_frame is None:
            self.prev_frame = next_frame
            self.prev_points = new_detected_points
            return new_detected_points
        new_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_frame, next_frame,
                                                             self.prev_points.astype(np.float32),
                                                             None, **lk_params)
        result = self.delta_fn(self.prev_points, new_detected_points, new_points)

        self.prev_points = result
        self.prev_frame = next_frame.copy()
        return result


class FilterTracker():
    def __init__(self):
        self.old_frame = None
        self.previous_landmarks_set = None
        self.with_landmark = True
        self.thres = 1.0
        self.alpha = 0.95
        self.iou_thres = 0.5
        self.filter = OneEuroFilter()

    def calculate(self, now_landmarks_set):
        if self.previous_landmarks_set is None or self.previous_landmarks_set.shape[0] == 0:
            self.previous_landmarks_set = now_landmarks_set
            result = now_landmarks_set
        else:
            if self.previous_landmarks_set.shape[0] == 0:
                return now_landmarks_set
            else:
                result = []
                for i in range(now_landmarks_set.shape[0]):
                    not_in_flag = True
                    for j in range(self.previous_landmarks_set.shape[0]):
                        if self.iou(now_landmarks_set[i], self.previous_landmarks_set[j]) > self.iou_thres:
                            result.append(self.smooth(now_landmarks_set[i], self.previous_landmarks_set[j]))
                            not_in_flag = False
                            break
                    if not_in_flag:
                        result.append(now_landmarks_set[i])
        result = np.array(result)
        self.previous_landmarks_set = result
        return result

    def iou(self, p_set0, p_set1):
        rec1 = [np.min(p_set0[:, 0]), np.min(p_set0[:, 1]), np.max(p_set0[:, 0]), np.max(p_set0[:, 1])]
        rec2 = [np.min(p_set1[:, 0]), np.min(p_set1[:, 1]), np.max(p_set1[:, 0]), np.max(p_set1[:, 1])]
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        # computing the sum_area
        sum_area = S_rec1 + S_rec2
        # find the each edge of intersect rectangle
        x1 = max(rec1[0], rec2[0])
        y1 = max(rec1[1], rec2[1])
        x2 = min(rec1[2], rec2[2])
        y2 = min(rec1[3], rec2[3])
        # judge if there is an intersect
        intersect = max(0, x2 - x1) * max(0, y2 - y1)
        return intersect / (sum_area - intersect)

    def smooth(self, now_landmarks, previous_landmarks):
        result = []
        for i in range(now_landmarks.shape[0]):
            dis = np.sqrt(np.square(now_landmarks[i][0] - previous_landmarks[i][0]) + np.square(
                now_landmarks[i][1] - previous_landmarks[i][1]))
            if dis < self.thres:
                result.append(previous_landmarks[i])
            else:
                result.append(self.filter(now_landmarks[i], previous_landmarks[i]))
        return np.array(result)

    def do_moving_average(self, p_now, p_previous):
        p = self.alpha * p_now + (1 - self.alpha) * p_previous
        return p


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.dx_prev = float(dx0)

    def __call__(self, x, x_prev):
        if x_prev is None:
            return x
        t_e = 1
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, x_prev)
        self.dx_prev = dx_hat
        return x_hat


class Tracker:
    def __init__(self):
        self.filter = FilterTracker()
        self.lk_tracker = LKTracker()

    def track(self, next_frame, landmarks):
        landmarks = self.lk_tracker.lk_track(next_frame, landmarks)
        landmarks = self.filter.calculate(np.array([landmarks]))[0]
        return landmarks
