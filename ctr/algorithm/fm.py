from __future__ import absolute_import
import random
import math
import cPickle
from ctr.common import utility


class FM(object):
    def __init__(self, total_features, k):
        self.total_features = total_features
        self.k = k
        self.w_0 = 0
        self.W = [0 for _ in xrange(self.total_features)]
        coef = 0.5 / math.sqrt(self.k)
        self.V = [coef * random.random() for _ in xrange(self.total_features * self.k)]

    def v_index(self, i, j):
        return i * self.k + j

    def train(self, feature_stream, _lambda, eta, init=False, report_interval=1000000):
        if init:
            self.w_0 = 0
            self.W = [0 for _ in xrange(self.total_features)]
            coef = 0.5 / math.sqrt(self.k)
            self.V = [coef * random.random() for _ in xrange(self.total_features * self.k)]
        validate_helper = utility.ValidateHelper(report_interval=report_interval)
        validate_helper.out_put()
        for count, (_, click, features) in enumerate(feature_stream):
            y = click > 0 and 1.0 or -1.0
            t = self.w_0
            field_count = len(features)
            sum_v_f = [0 for _ in xrange(self.k)]
            for f_index_1 in xrange(field_count):
                f1 = features[f_index_1]
                t += self.W[f1]
                for f_index_2 in xrange(f_index_1 + 1, field_count):
                    f2 = features[f_index_2]
                    for f in xrange(1, self.k):
                        v1 = self.V[self.v_index(f1, f)]
                        v2 = self.V[self.v_index(f2, f)]
                        t += v1 * v2
                for f in xrange(self.k):
                    sum_v_f[f] += self.V[self.v_index(f1, f)]
            p = utility.sigmoid(t)
            const_part = (utility.sigmoid(t*y) - 1) * y
            self.w_0 -= eta * (const_part + _lambda * self.w_0)
            for feature_index in features:
                self.W[feature_index] -= eta * (const_part + _lambda * self.W[feature_index])
                for f in xrange(self.k):
                    v_index = self.v_index(feature_index, f)
                    v = self.V[v_index]
                    self.V[v_index] -= eta * (const_part * (sum_v_f[f] - v) + _lambda * self.V[v_index])
            if p >= 1:
                print p, t
            validate_helper.update(p, click, 0.5)
        validate_helper.out_put()
        return validate_helper.get_log_loss()

    def test(self, feature_stream, p_threshold=0.5,report_interval=1000000):
        validate_helper = utility.ValidateHelper(report_interval=report_interval)
        for count, (_, click, features) in enumerate(feature_stream):
            t = self.w_0
            field_count = len(features)
            sum_v_f = [0 for _ in xrange(self.k)]
            for f_index_1 in xrange(field_count):
                f1 = features[f_index_1]
                t += self.W[f1]
                for f_index_2 in xrange(f_index_1 + 1, field_count):
                    f2 = features[f_index_2]
                    for f in xrange(1, self.k):
                        v1 = self.V[self.v_index(f1, f)]
                        v2 = self.V[self.v_index(f2, f)]
                        t += v1 * v2
                for f in xrange(self.k):
                    sum_v_f[f] += self.V[self.v_index(f1, f)]
            p = utility.sigmoid(t)
            validate_helper.update(p, click, p_threshold)
        validate_helper.out_put()
        return validate_helper.get_log_loss()

    def dump_model(self, filename):
        with open(filename, "wb") as f:
            cPickle.dump(self.__dict__, f)

    def load_model(self, filename):
        with open(filename, "rb") as f:
            self.__dict__ = cPickle.load(f)