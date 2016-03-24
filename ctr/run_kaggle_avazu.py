from __future__ import absolute_import
from ctr.algorithm import fctl
from ctr.common import utility

if __name__ == "__main__":
    train_fs = utility.FeatureStream(utility.get_date_file_path("app_feature_map_v2.json"), utility.get_date_file_path("app_train_features_v2.csv"))
    print "training......"
    alg = fctl.Fctl(train_fs.feature_count())
    #alg.load_model(utility.get_date_file_path("app_model_1.txt"))
    for i in range(5):
        train_fs.reset()
        print "iter {0}......".format(i)
        alg.train(train_fs, 0.1, 1, 0.001, 0.001)
        alg.dump_model(utility.get_date_file_path("app_model_1.txt"))
        alg.dump_model_readable(train_fs.feature_map, utility.get_date_file_path("app_model_1_readable.txt"))
        print "testing......"
        test_fs = utility.FeatureStream(utility.get_date_file_path("app_feature_map_v2.json"), utility.get_date_file_path("app_test_features_v2.csv"))
        alg.test(test_fs)
