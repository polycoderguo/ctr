from __future__ import absolute_import
from ctr.common import utility
import numpy as np
import os
import cPickle
import sys

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
    t = 1.0 / (1.0 + np.exp(-inX))
    return t


def ftrl(alpha, beta, lamba1, lamba2, fs, feature_map, model_file):
    z = np.zeros(len(feature_map))
    n = np.zeros(len(feature_map))
    w = np.zeros(len(feature_map))
    logloss_counter = utility.LogLossCounter()
    for count, (click, features) in enumerate(fs):
        x = np.zeros((1, len(feature_map)))
        no_zero_index = []
        for feature in features:
            try:
                feature_index = feature_map[int(feature)]
                no_zero_index.append(feature_index)
                x[0, feature_index] = 1
                if np.abs(z[feature_index]) > lamba1:
                    t = (-1.0 / ((beta + np.sqrt(n[feature_index])) / alpha + lamba2)) * (z[feature_index] - np.sign(z[feature_index]) * lamba1)
                    w[feature_index] = t
                else:
                    w[feature_index] = 0
            except Exception as e:
                continue
        p = sigmoid(x.dot(w)[0])
        logloss_counter.count_logloss(p, click)
        for feature_index in no_zero_index:
            g = p - click
            sigma = (1.0 / alpha) * (np.sqrt(n[feature_index] + np.exp2(g)) - np.sqrt(n[feature_index]))
            w_i = w[feature_index]
            z[feature_index] += g - sigma * w_i
            n[feature_index] += np.exp2(g)
    with open(model_file, "wb") as f:
        f.write(cPickle.dumps(w))

if __name__ == "__main__":
    root = os.path.dirname(__file__)
    feature_map = load_feature_map(os.path.join(root, "../data/app_feature_map.txt"))
    fs = TrainStream(os.path.join(root, "../data/app_train_features.txt"))
    try:
        alpha, beta, lamba1, lamba2, model_file = sys.argv[1:]
    except:
        alpha, beta, lamba1, lamba2, model_file = 8, 1, 0.001, 0.001, "model.txt"
    ftrl(float(alpha), float(beta), float(lamba1), float(lamba2), fs, feature_map, os.path.join(root, "../data/{0}").format(model_file))