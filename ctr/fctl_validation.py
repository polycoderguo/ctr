from __future__ import absolute_import
from ctr.fctl import load_feature_map, TrainStream, sigmoid
import numpy as np
import os
import cPickle
from ctr.common import utility


def validation(map_file, modelfile, validate_file):
    feature_map = load_feature_map(map_file)
    w = cPickle.load(open(model_file, "rb"))
    vs = TrainStream(validate_file)
    logloss_counter = utility.LogLossCounter()
    for count, (click, features) in enumerate(vs):
        x = np.zeros((1, len(feature_map)))
        for feature in features:
            try:
                feature_index = feature_map[int(feature)]
                x[0, feature_index] = 1
            except Exception as e:
                continue
        p = sigmoid(x.dot(w)[0])
        p = 0.14
        logloss_counter.count_logloss(p, click)
    logloss_counter.output()

if __name__ == "__main__":
    root = os.path.dirname(__file__)
    map_file = os.path.join(root, "../data/site_feature_map.txt")
    validate_file = os.path.join(root, "../data/site_validate_features.txt")
    model_file = os.path.join(root, "../data/model01-site.txt")
    validation(map_file, model_file, validate_file)