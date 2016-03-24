from __future__ import absolute_import
import numpy as np
from ctr.common import math
from ctr.common import utility
import cPickle


class Fctl(object):
    def __init__(self, feature_count):
        self.feature_count = feature_count
        self.z = np.zeros(self.feature_count)
        self.n = np.zeros(self.feature_count)
        self.w = np.zeros(self.feature_count)

    def train(self, feature_stream, alpha, beta, lamba1, lamba2, init=False):
        if init:
            self.z = np.zeros(self.feature_count)
            self.n = np.zeros(self.feature_count)
            self.w = np.zeros(self.feature_count)
        validate_helper = utility.ValidateHelper()
        validate_helper.out_put()
        for count, (click, features) in enumerate(feature_stream):
            t = 0
            for feature_index in features:
                if np.abs(self.z[feature_index]) > lamba1:
                    _t = (-1.0 / ((beta + np.sqrt(self.n[feature_index])) / alpha + lamba2)) * (self.z[feature_index] - np.sign(self.z[feature_index]) * lamba1)
                    self.w[feature_index] = _t
                    t += _t
                else:
                    self.w[feature_index] = 0
            p = math.sigmoid(t)
            for feature_index in features:
                g = p - click
                sigma = (1.0 / alpha) * (np.sqrt(self.n[feature_index] + g*g) - np.sqrt(self.n[feature_index]))
                self.z[feature_index] += g - sigma * self.w[feature_index]
                self.n[feature_index] += g*g
            validate_helper.update(p, click, 0.5)
        validate_helper.out_put()

    def test(self, feature_stream, p_threshold=0.5):
        validate_helper = utility.ValidateHelper()
        for count, (click, features) in enumerate(feature_stream):
            t = 0
            for feature_index in features:
                t += self.w[feature_index]
            p = math.sigmoid(t)
            validate_helper.update(p, click, p_threshold)
        validate_helper.out_put()

    def dump_model(self, filename):
        with open(filename, "wb") as f:
            cPickle.dump(self.__dict__, f)

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.__dict__ = cPickle.load(f)

    def dump_model_readable(self, feature_map, filename):
        feature_weights = []
        for i in range(self.feature_count):
            if self.w[i] > 0:
                feature_weights.append((self.w[i], i, feature_map.feature_index_2_str(i)))
        feature_weights.sort()
        feature_weights.reverse()
        with open(filename, "wb") as f:
            for weight, index, str_feature in feature_weights:
                f.write("{0}\t{1}\t{2}\r\n".format(index, str_feature, weight))
