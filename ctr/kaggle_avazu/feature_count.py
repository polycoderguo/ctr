from __future__ import absolute_import
from ctr.common import utility

if __name__ == "__main__":
    train_file = utility.get_date_file_path("t.csv")
    utility.count_csv_values_by_field(train_file, utility.get_date_file_path("features"))
