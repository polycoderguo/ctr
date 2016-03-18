from __future__ import absolute_import
from ctr.common import utility
from ctr.common.utility import CSVReader
from collections import defaultdict


def convert_feature(train_file, feature_file, map_file):
    reader = CSVReader(train_file)
    device_ip_count = defaultdict(int)
    device_id_count = defaultdict(int)
    smooth_user_hour_count = defaultdict(int)
    user_count = defaultdict(int)
    user_hour_count = defaultdict(int)
    while True:
        row = reader.read_row()
        if row is None:
            break
        site_id = reader.read_row_data(row, "site_id")
        is_app = False
        if utility.is_app(site_id):
            is_app = True
            pub_id = reader.read_row_data(row, "app_id")
            pub_domain = reader.read_row_data(row, "app_domain")
            pub_category = reader.read_row_data(row, "app_category")
        else:
            pub_id = reader.read_row_data(row, "site_id")
            pub_domain = reader.read_row_data(row, "site_domain")
            pub_category = reader.read_row_data(row, "site_category")
        banner_pos = reader.read_row_data(row, "banner_pos")
        device_model = reader.read_row_data(row, "device_model")
        device_conn_type = reader.read_row_data(row, "device_conn_type")
        C14, C17, C20, C21 = reader.read_row_data(row, "C14"), reader.read_row_data(row, "C17"), reader.read_row_data(row, "C20"), reader.read_row_data(row, "C21")
        hour = reader.read_row_data(row, "hour")
        device_id = reader.read_row_data(row, "device_id")
        device_id_count[device_id] += 1
        device_ip = reader.read_row_data(row,  "device_ip")
        device_ip_count[device_ip] += 1
        if utility.has_id_info(device_id):
            user_id = device_id
        else:
            user_id = device_ip + device_model
        user_count[user_id] += 1
        user_hour_count[user_id + '-' + hour] += 1
        utility.counting_line(reader.index)


if __name__ == "__main__":
    import os
    root = os.path.dirname(__file__)
    convert_feature(os.path.join(root, "../data/t.csv"), os.path.join(root, "../data/train_features.txt"), os.path.join(root, "../data/feature_map.txt"))