from __future__ import absolute_import
from ctr.algorithm import fm
from ctr.common import utility
import sys

if __name__ == "__main__":
    feature_map_file = "app_feature_map.json"
    train_data_file = "app_train_features.csv"
    test_data_file = "app_test_features.csv"
    if len(sys.argv) > 2:
        eta, _lambda, k, iter = sys.argv[1:]
        eta, _lambda, k, iter = float(eta), float(_lambda), int(k), int(iter)
    else:
        eta, _lambda, iter = 0.03, 0.00002, 4, 15
    train_fs = utility.FeatureStream(utility.get_date_file_path(feature_map_file), utility.get_date_file_path(train_data_file))
    model_file = "app_model_{0}_{1}.json".format(eta, _lambda)
    print "training......"
    alg = fm.FM(train_fs.feature_count(), k)
    for i in xrange(iter):
        train_fs.reset()
        print "iter {0}......".format(i)
        #alg.load_model(utility.get_date_file_path(model_file))
        alg.train(train_fs, _lambda, eta)
        alg.dump_model(utility.get_date_file_path(model_file))
        print "testing......"
        test_fs = utility.FeatureStream(utility.get_date_file_path(feature_map_file), utility.get_date_file_path(test_data_file))
        alg.test(test_fs)
