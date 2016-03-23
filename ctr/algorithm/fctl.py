from __future__ import absolute_import
import numpy as np
from ctr.common import math
from ctr.common import utility
import cPickle


class Fctl(object):
    def __init__(self, feature_count):
        self.feature_count = feature_count
        self.z = None
        self.n = None
        self.w = None

    def train(self, feature_stream, alpha, beta, lamba1, lamba2):
        validate_helper = utility.ValidateHelper()
        self.z = np.zeros(self.feature_count)
        self.n = np.zeros(self.feature_count)
        self.w = np.zeros(self.feature_count)
        for count, (click, features) in enumerate(feature_stream):
            no_zero_index = []
            t = 0
            for feature_index in features:
                no_zero_index.append(feature_index)
                if np.abs(self.z[feature_index]) > lamba1:
                    _t = (-1.0 / ((beta + np.sqrt(self.n[feature_index])) / alpha + lamba2)) * (self.z[feature_index] - np.sign(self.z[feature_index]) * lamba1)
                    self.w[feature_index] = _t
                    t += _t
                else:
                    self.w[feature_index] = 0
            p = math.sigmoid(t)
            for feature_index in no_zero_index:
                g = p - click
                sigma = (1.0 / alpha) * (np.sqrt(self.n[feature_index] + np.exp2(g)) - np.sqrt(self.n[feature_index]))
                w_i = self.w[feature_index]
                self.z[feature_index] += g - sigma * w_i
                self.n[feature_index] += np.exp2(g)
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

