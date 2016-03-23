from __future__ import absolute_import
import sys
from ctr.algorithm import fctl
from ctr.common import utility

if __name__ == "__main__":
    train_fs = utility.FeatureStream(utility.get_date_file_path("app_feature_map.json"), utility.get_date_file_path("app_train_features.csv"))
    print "training......"
    alg = fctl.Fctl(train_fs.feature_count())
    alg.train(train_fs, 0.1, 1, 0.01, 0.01)
    alg.dump_model(utility.get_date_file_path("app_model_1.txt"))
    print "testing......"
    alg.load_model(utility.get_date_file_path("app_model_1.txt"))
    test_fs = utility.FeatureStream(utility.get_date_file_path("app_feature_map.json"), utility.get_date_file_path("app_test_features.csv"))
    alg.test(test_fs)
