import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None):
        super(PriorBox, self).__init__()
        anchors = [[[20.816779525064888, 20.69631680086565], [30.797590071368038, 26.392322382537383],
                    [29.551720198080584, 39.24676118593017], [39.98885017493792, 33.29985790877724]],
                   [[48.99104888069727, 42.37459569432322], [40.94755241973759, 51.45702253627861],
                    [56.34899448908407, 49.003406381748206], [48.005429425330775, 62.33321923292782]],
                   [[64.26108673168943, 54.204093529199014], [61.49837256658091, 63.37901611185405],
                    [56.8604260921569, 75.76107730332906], [72.321645976738, 61.644766857420485]],
                   [[71.49620817590952, 75.8112036180628], [81.75913728562372, 68.66644621740605],
                    [89.23855201653036, 86.92088644095563], [100.40889496147852, 98.01200582532299]]]

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

