from __future__ import absolute_import
from ctr.algorithm import fctl
from ctr.common import utility
import sys

if __name__ == "__main__":
    feature_map_file = "app_feature_map_v2.json"
    train_data_file = "app_train_features_v2.csv"
    test_data_file = "app_test_features_v2.csv"
    if len(sys.argv) > 2:
        alpha, beta, lambda1, lambda2, iter = sys.argv[1:]
        alpha, beta, lambda1, lambda2, iter = float(alpha), float(beta), float(lambda1), float(lambda2), int(iter)
    else:
        alpha, beta, lambda1, lambda2, iter = 1, 1, 0.001, 0.001, 10
    train_fs = utility.FeatureStream(utility.get_date_file_path(feature_map_file), utility.get_date_file_path(train_data_file))
    model_file = "app_model_fctl_v2_{0}_{1}_{2}_{3}.json".format(alpha, beta, lambda1, lambda2)
    readable_model_file = "app_model_fctl_v2_{0}_{1}_{2}_{3}_readable.json".format(alpha, beta, lambda1, lambda2)
    print "training......"
    alg = fctl.Fctl(train_fs.feature_count())
    #alg.load_model(utility.get_date_file_path("app_model_1.txt"))
    for i in xrange(iter):
        train_fs.reset()
        print "iter {0}......".format(i)
        alg.train(train_fs, alpha, beta, lambda1, lambda2)
        alg.dump_model(utility.get_date_file_path(model_file))
        alg.dump_model_readable(train_fs.feature_map, utility.get_date_file_path(readable_model_file))
        print "testing......"
        test_fs = utility.FeatureStream(utility.get_date_file_path(feature_map_file), utility.get_date_file_path(test_data_file))
        alg.test(test_fs)
