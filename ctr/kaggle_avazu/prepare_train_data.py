from __future__ import absolute_import
from ctr.common import utility

if __name__ == "__main__":
    utility.split_train_test_data_set(utility.get_date_file_path("train.csv"), utility.get_date_file_path("t.csv"), utility.get_date_file_path("v.csv"), test_samples_rate=0.1)