from __future__ import absolute_import
from ctr.common import utility
import numpy as np
import os


def load_feature_map(fname):
    features = []
    with open(fname, "rb") as f:
        for count, line in enumerate(f):
            try:
                t = line.strip().split("\t")
                features.append(int(t[0]))
            except:
                pass
    feature_map = {}
    for count, feature in enumerate(features):
        feature_map[feature] = count
    return feature_map


class TrainStream(object):
    def __init__(self, filename):
        self.f = open(filename, "rb")

    def __iter__(self):
        return self

    def next(self):
        try:
            t = self.f.readline().strip().split(',')
            if len(t) == 0:
                raise StopIteration()
            return int(t[0]), t[1:]
        except:
            raise StopIteration()


def sigmoid(inX):
    t = 1.0 /(1+ np.exp(-inX))
    return t


def ftrl(alpha, beta, lamba1, lamba2, fs, feature_map):
    z = np.zeros((1, len(feature_map)))
    n = np.zeros((1, len(feature_map)))
    for count, (click, features) in enumerate(fs):
        x = np.zeros((1, len(feature_map)))
        w = np.zeros((len(feature_map), 1))
        no_zero_index = []
        for feature in features:
            try:
                feature_index = feature_map[int(feature)]
                no_zero_index.append(feature_index)
                x[0, feature_index] = 1
                if z[0, feature_index] > lamba1:
                    t = (-1 / ((beta + np.sqrt(n[0, feature_index])) / alpha + lamba2)) * (z[0, feature_index] - np.sign(z[0, feature_index]) * lamba1)
                    w[feature_index, 0] = t
                    pass
            except Exception as e:
                continue
        p = sigmoid(x.dot(w)[0, 0])
        if p > 0.4:
            print p, click
        for feature_index in no_zero_index:
            g = p - click
            sigma = (1 / alpha) * (np.sqrt(n[0, feature_index] + np.exp2(g)) - np.sqrt(n[0, feature_index]))
            w_i = w[feature_index, 0]
            z[0, feature_index] += g - sigma * w_i
            n[0, feature_index] += np.exp2(g)
        utility.counting_line(count, 100000)

if __name__ == "__main__":
    root = os.path.dirname(__file__)
    feature_map = load_feature_map(os.path.join(root, "../data/feature_map.txt"))
    fs = TrainStream(os.path.join(root, "../data/train_features.txt"))
    import time
    t = time.time()
    ftrl(1, 1, 0.1, 0.1, fs, feature_map)
    print time.time() - t