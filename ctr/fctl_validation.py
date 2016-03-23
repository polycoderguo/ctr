from __future__ import absolute_import
from ctr.fctl import load_feature_map, TrainStream, sigmoid
import numpy as np
import os
import cPickle
from ctr.common import utility


def validation(map_file, modelfile, validate_file):
    feature_map = load_feature_map(map_file)
    w = cPickle.load(open(modelfile, "rb"))
    vs = TrainStream(validate_file)
    logloss_counter = utility.LogLossCounter()
    total = 0
    clicked = 0
    p_clicked = 0
    p_clicked_correct = 0
    for count, (click, features) in enumerate(vs):
        x = np.zeros((1, len(feature_map)))
        for feature in features:
            try:
                feature_index = feature_map[int(feature)]
                x[0, feature_index] = 1
            except Exception as e:
                continue
        total += 1
        p = sigmoid(x.dot(w)[0])
        if p > 0.9:
            p_clicked += 1
            if click == 1:
                p_clicked_correct += 1
        if click == 1:
            clicked += 1
        logloss_counter.count_logloss(p, click)
    logloss_counter.output()
    print total, clicked, p_clicked, p_clicked_correct

if __name__ == "__main__":
    root = os.path.dirname(__file__)
    map_file = os.path.join(root, "../data/app_feature_map.txt")
    validate_file = os.path.join(root, "../data/app_validate_features.txt")
    model_file = os.path.join(root, "../data/model1.txt")
    validation(map_file, model_file, validate_file)