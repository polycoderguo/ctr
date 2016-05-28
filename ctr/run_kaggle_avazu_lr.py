from __future__ import absolute_import
from ctr.algorithm import lr
from ctr.common import utility
import sys

if __name__ == "__main__":
    feature_map_file = "app_feature_map.json"
    train_data_file = "app_train_features.csv"
    test_data_file = "app_test_features.csv"
    if len(sys.argv) > 2:
        eta, _lambda, iter = sys.argv[1:]
        eta, _lambda, iter = float(eta), int(iter)
    else:
        eta, _lambda, iter = 0.003, 0, 20
    feature_map = utility.DummyFeatureMap.load(utility.get_date_file_path(feature_map_file))
    train_fs = utility.FeatureStream(feature_map, utility.get_date_file_path(train_data_file))
    print "training......"
    alg = lr.LR(train_fs.feature_count())
    #alg.load_model(utility.get_date_file_path("app_model_1.txt"))
    for i in xrange(iter):
        train_fs.reset()
        model_file = "app_model_lr_v3_{0}_{1}.json".format(eta, iter)
        print "iter {0}......".format(i)
        alg.train(train_fs, _lambda, eta, report_interval=1000000)
        alg.dump_model(utility.get_date_file_path(model_file))
        print "testing......"
        test_fs = utility.FeatureStream(utility.get_date_file_path(feature_map_file), utility.get_date_file_path(test_data_file))
        alg.test(test_fs)
