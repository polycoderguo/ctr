from __future__ import absolute_import
import numpy as np
from ctr.common import utility
import cPickle


class LR(object):
    def __init__(self, total_features):
        self.total_features = total_features
        self.W = np.zeros(total_features + 1)

    def train(self, feature_stream, _lambda, eta, init=False, report_interval=1000000):
        if init:
            self.W = np.zeros(self.total_features + 1)
        validate_helper = utility.ValidateHelper(report_interval=report_interval)
        validate_helper.out_put()
        for count, (_, click, features) in enumerate(feature_stream):
            t = self.W[0]
            for feature_index in features:
                t += self.W[feature_index]
            p = utility.sigmoid(t)
            g = eta * (p - click)
            self.W[0] -= g
            for feature_index in features:
                self.W[feature_index] -= (g + _lambda * self.W[feature_index])
            validate_helper.update(p, click, 0.5)
        validate_helper.out_put()

    def test(self, feature_stream, p_threshold=0.5, report_interval=1000000):
        validate_helper = utility.ValidateHelper(report_interval=report_interval)
        for count, (_, click, features) in enumerate(feature_stream):
            t = self.W[0]
            for feature_index in features:
                t += self.W[feature_index]
            p = utility.sigmoid(t)
            validate_helper.update(p, click, p_threshold)
        validate_helper.out_put()

    def dump_model(self, filename):
        with open(filename, "wb") as f:
            cPickle.dump(self.__dict__, f)

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.__dict__ = cPickle.load(f)