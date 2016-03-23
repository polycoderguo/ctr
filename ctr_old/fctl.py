from __future__ import absolute_import
from ctr_old.common import utility
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
    def __init__(self, filename, start_lines = 0, max_lines = -1):
        self.f = open(filename, "rb")
        for i in xrange(start_lines):
            self.f.readline()
        self.count = 0
        self.max_lines = max_lines

    def __iter__(self):
        return self

    def next(self):
        if 0 < self.max_lines <= self.count:
            raise StopIteration()
        try:
            t = self.f.readline().strip().split(',')
            if len(t) == 0:
                raise StopIteration()
            self.count += 1
            return int(t[0]), t[1:]
        except:
            raise StopIteration()


def sigmoid(inX):
    t = 1.0 / (1.0 + np.exp(-inX))
    return t


def ftrl(alpha, beta, lamba1, lamba2, train_fs, validate_fs, feature_map, model_file):
    z = np.zeros(len(feature_map))
    n = np.zeros(len(feature_map))
    w = np.zeros(len(feature_map))
    logloss_counter = utility.LogLossCounter()
    for count, (click, features) in enumerate(train_fs):
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
    logloss_counter.output()
    print "start validation......"
    logloss_counter = utility.LogLossCounter()
    for count, (click, features) in enumerate(validate_fs):
        x = np.zeros((1, len(feature_map)))
        for feature in features:
            try:
                feature_index = feature_map[int(feature)]
                x[0, feature_index] = 1
            except Exception as e:
                continue
        p = sigmoid(x.dot(w)[0])
        logloss_counter.count_logloss(p, click)
    logloss_counter.output()
    with open(model_file, "wb") as f:
        f.write(cPickle.dumps(w))

if __name__ == "__main__":
    root = os.path.dirname(__file__)
    map_file = os.path.join(root, "../data/app_feature_map.txt")
    train_file = os.path.join(root, "../data/app_train_features.txt")
    total_samples = utility.count_file_lines(train_file)
    train_samples = int(total_samples * 0.8)
    feature_map = load_feature_map(map_file)
    fs = TrainStream(train_file)
    vs = TrainStream(train_file, start_lines=train_samples)
    try:
        alpha, beta, lamba1, lamba2, model_file = sys.argv[1:]
    except:
        alpha, beta, lamba1, lamba2, model_file = 1, 1, 0.001, 0.001, "model.txt"
    ftrl(float(alpha), float(beta), float(lamba1), float(lamba2), fs, vs, feature_map, os.path.join(root, "../data/{0}").format(model_file))