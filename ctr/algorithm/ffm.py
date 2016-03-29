from __future__ import absolute_import
import random
import math
import cPickle
from ctr.common import utility


class FFM(object):
    def __init__(self, total_features, total_fields, k):
        self.total_features = total_features
        self.total_fields = total_fields
        self.k = k
        self.v = 2.0 / float(self.k)
        self.W = [0.0 for _ in xrange(self.total_features * self.total_fields * self.k)]
        self.W2 = [1.0 for _ in xrange(self.total_features * self.total_fields * self.k)]
        coef = 0.5 / math.sqrt(self.k)
        for j in xrange(self.total_features):
            for f in xrange(self.total_fields):
                for d in xrange(self.k):
                    self.W[self.W_index(j, f, d)] = coef * random.random()

    def W_index(self, j, f, d):
        return j * (self.total_fields * self.k) + f * self.k + d

    def train(self, feature_stream, _lambda, eta, init=False, report_interval=1000000):
        if init:
            self.W = [0.0 for _ in xrange(self.total_features * self.total_fields * self.k)]
            self.W2 = [1.0 for _ in xrange(self.total_features * self.total_fields * self.k)]
            coef = 0.5 / math.sqrt(self.k)
            self.v = 2.0 / float(self.k)
            for j in xrange(self.total_features):
                for f in xrange(self.total_fields):
                    for d in xrange(self.k):
                        self.W[self.W_index(j, f, d)] = coef * random.random()
        validate_helper = utility.ValidateHelper(report_interval=report_interval)
        validate_helper.out_put()
        align_0 = self.k
        align_1 = self.total_fields * align_0
        for count, (_, click, features) in enumerate(feature_stream):
            y = click > 0 and 1.0 or -1.0
            t = 0.0
            field_count = len(features)
            for f_index_1 in xrange(field_count):
                j1 = features[f_index_1]
                for f_index_2 in xrange(f_index_1 + 1, field_count):
                    j2 = features[f_index_2]
                    j1_align = j1 * align_1 + f_index_2 * align_0
                    j2_align = j2 * align_1 + f_index_1 * align_0
                    for d in xrange(self.k):
                        #t += (self.W[self.W_index(j1, f_index_1, d)] * self.W[self.W_index(j2, f_index_2, d)] * self.v)
                        t += (self.W[j1_align + d] * self.W[j2_align + d] * self.v)
            p = utility.sigmoid(t)
            expnyt = math.exp(-y*t)
            kappav = (-y*expnyt/(1+expnyt)) * self.v
            for f_index_1 in xrange(field_count):
                j1 = features[f_index_1]
                for f_index_2 in xrange(f_index_1 + 1, field_count):
                    j2 = features[f_index_2]
                    j1_align = j1 * align_1 + f_index_2 * align_0
                    j2_align = j2 * align_1 + f_index_1 * align_0
                    for d in xrange(self.k):
                        w1_index = j1_align + d
                        w2_index = j2_align + d
                        g1 = _lambda * self.W[w1_index] + kappav * self.W[w2_index]
                        g2 = _lambda * self.W[w2_index] + kappav * self.W[w1_index]
                        self.W2[w1_index] += g1*g1
                        self.W2[w2_index] += g2*g2
                        self.W[w1_index] -= eta * (1.0 / math.sqrt(self.W2[w1_index])) * g1
                        self.W[w2_index] -= eta * (1.0 / math.sqrt(self.W2[w2_index])) * g2
            validate_helper.update(p, click, 0.5)
        validate_helper.out_put()
        return validate_helper.get_log_loss()

    def predict(self, features):
        t = 0.0
        field_count = len(features)
        align_0 = self.k
        align_1 = self.total_fields * align_0
        for f_index_1 in xrange(field_count):
            j1 = features[f_index_1]
            if j1 < 0 or j1 >= self.total_features:
                continue
            for f_index_2 in xrange(f_index_1 + 1, field_count):
                j2 = features[f_index_2]
                if j2 < 0 or j2 >= self.total_features:
                    continue
                j1_align = j1 * align_1 + f_index_2 * align_0
                j2_align = j2 * align_1 + f_index_1 * align_0
                for d in xrange(self.k):
                    #t += (self.W[self.W_index(j1, f_index_1, d)] * self.W[self.W_index(j2, f_index_2, d)] * self.v)
                    t += (self.W[j1_align + d] * self.W[j2_align + d] * self.v)
        p = utility.sigmoid(t)
        return p

    def test(self, feature_stream, p_threshold=0.5,report_interval=1000000):
        validate_helper = utility.ValidateHelper(report_interval=report_interval)
        align_0 = self.k
        align_1 = self.total_fields * align_0
        for count, (_, click, features) in enumerate(feature_stream):
            t = 0.0
            field_count = len(features)
            for f_index_1 in xrange(field_count):
                j1 = features[f_index_1]
                for f_index_2 in xrange(f_index_1 + 1, field_count):
                    j2 = features[f_index_2]
                    j1_align = j1 * align_1 + f_index_2 * align_0
                    j2_align = j2 * align_1 + f_index_1 * align_0
                    for d in xrange(self.k):
                        #t += (self.W[self.W_index(j1, f_index_1, d)] * self.W[self.W_index(j2, f_index_2, d)] * self.v)
                        t += (self.W[j1_align + d] * self.W[j2_align + d] * self.v)
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