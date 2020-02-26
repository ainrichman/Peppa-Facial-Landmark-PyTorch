import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None):
        super(PriorBox, self).__init__()
        anchors = [[[35.3819, 46.9085], [40.5289, 55.4810], [44.6852, 62.8607]],
                   [[49.7344, 71.2506], [58.1781, 80.5151], [63.9462, 90.5969]],
                   [[69.8191, 83.7008], [70.8205, 98.5648], [77.2793, 110.2457]]]
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.variance = cfg['variance']
        self.image_size = image_size
        self.feature_maps = cfg['feature_maps']
        self.anchors = np.array(anchors)

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            anchors = self.anchors[k]
            f_w = f[0]
            f_h = f[1]
            for i, j in product(range(f_w), range(f_h)):
                for anchor in anchors:
                    w = anchor[0] / self.image_size[0]
                    h = anchor[1] / self.image_size[1]
                    mean += [(i + 0.5) / f_w, (j + 0.5) / f_h, w, h]
        output = np.array(mean).reshape((-1, 4))
        if self.clip:
            output.clamp_(max=1, min=0)
        return output.astype(np.float32)
