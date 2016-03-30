from __future__ import absolute_import
from ctr.algorithm import ffm
from ctr.kaggle_avazu.feature_engine_v3 import *

if __name__ == "__main__":
    feature_map_file = utility.get_date_file_path("feature_map_v3_submit.json")
    train_feature_file = utility.get_date_file_path("train_features_v3_submit.csv")
    app_train_feature_file = utility.get_date_file_path("app_train_features_v3_submit.csv")
    site_train_feature_file = utility.get_date_file_path("site_train_features_v3_submit.csv")
    test_feature_file = utility.get_date_file_path("test_features_v3_submit.csv")
    app_model_file = utility.get_date_file_path("app_model_v3_submit.csv")
    site_model_file = utility.get_date_file_path("site_model_v3_submit.csv")

    train_data_file = utility.get_date_file_path("train.csv")
    test_data_file = utility.get_date_file_path("test.csv")

    submit_file_name = utility.get_date_file_path("submit.csv")

    print "Scan......"
    scan(train_data_file)
    scan(test_data_file)
    print "Train features....."
    convert_feature(train_data_file, train_feature_file, feature_map_file)
    print "Test features....."
    convert_feature(test_data_file, test_feature_file, feature_map_file, shared_map_file=feature_map_file, is_train=False, submit=True)

    eta, _lambda, k, iter = 0.03, 0.0002, 4, 13
    train_fs = utility.FeatureStream(feature_map_file, app_train_feature_file)
    alg = ffm.FFM(train_fs.feature_count(), train_fs.field_count(), k)
    print "Start learning app......"
    for i in xrange(iter):
        train_fs.reset()
        print "iter {0}......".format(i)
        alg.train(train_fs, _lambda, eta, report_interval=-1)
        alg.dump_model(app_model_file)

    eta, _lambda, k, iter = 0.03, 0.0002, 4, 17
    train_fs = utility.FeatureStream(feature_map_file, site_train_feature_file)
    alg = ffm.FFM(train_fs.feature_count(), train_fs.field_count(), k)
    print "Start learning site......"
    for i in xrange(iter):
        train_fs.reset()
        print "iter {0}......".format(i)
        alg.train(train_fs, _lambda, eta, report_interval=-1)
        alg.dump_model(site_model_file)

    print "Start predict......"
    test_fs = utility.FeatureStream(feature_map_file, train_data_file, click_at=1)
    alg_app = ffm.FFM(train_fs.feature_count(), train_fs.field_count(), k)
    alg_app.load_model(app_model_file)
    alg_site = ffm.FFM(train_fs.feature_count(), train_fs.field_count(), k)
    alg_site.load_model(site_model_file)
    with open(submit_file_name, "rb") as f:
        for count, ((id,), app_row, features) in enumerate(test_fs):
            if app_row == "1":
                p = alg_app.predict(features)
            else:
                p = alg_site.predict(features)
            f.write("%s,%.10f\r\n".format(id, p))
            utility.progress(count)
