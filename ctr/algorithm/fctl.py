from __future__ import absolute_import
from ctr.common import utility
import cPickle
import math


class Fctl(object):
    def __init__(self, feature_count):
        self.feature_count = feature_count
        self.z = [0 for _ in xrange(self.feature_count)]
        self.n = [0 for _ in xrange(self.feature_count)]
        self.w = [0 for _ in xrange(self.feature_count)]

    def train(self, feature_stream, alpha, beta, lamba1, lamba2, init=False, report_interval=1000000):
        if init:
            self.z = [0 for _ in xrange(self.feature_count)]
            self.n = [0 for _ in xrange(self.feature_count)]
            self.w = [0 for _ in xrange(self.feature_count)]
        validate_helper = utility.ValidateHelper(report_interval=report_interval)
        validate_helper.out_put()
        for count, (click, features) in enumerate(feature_stream):
            t = 0
            for feature_index in features:
                if abs(self.z[feature_index]) > lamba1:
                    _t = (-1.0 / ((beta + math.sqrt(self.n[feature_index])) / alpha + lamba2)) * (self.z[feature_index] - utility.sign(self.z[feature_index]) * lamba1)
                    self.w[feature_index] = _t
                    t += _t
                else:
                    self.w[feature_index] = 0
            p = utility.sigmoid(t)
            for feature_index in features:
                g = p - click
                sigma = (1.0 / alpha) * (math.sqrt(self.n[feature_index] + g*g) - math.sqrt(self.n[feature_index]))
                self.z[feature_index] += g - sigma * self.w[feature_index]
                self.n[feature_index] += g*g
            validate_helper.update(p, click, 0.5)
        validate_helper.out_put()

    def test(self, feature_stream, p_threshold=0.5, report_interval=1000000):
        validate_helper = utility.ValidateHelper(report_interval=report_interval)
        for count, (click, features) in enumerate(feature_stream):
            t = 0
            for feature_index in features:
                t += self.w[feature_index]
            p = utility.sigmoid(t)
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
