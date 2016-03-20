from __future__ import absolute_import
from ctr.common import utility
from ctr.common.utility import CSVReader
from collections import defaultdict


def map_features(features, feature_map):
    t = []
    for f in features:
        sf = utility.hashstr(f)
        try:
            feature_map[sf][1] += 1
        except:
            feature_map[sf] = [f, 1]
        t.append(sf)
    return ",".join(t)


def convert_feature(train_file, feature_file, map_file):
    reader = CSVReader(train_file)
    device_ip_count = defaultdict(int)
    device_id_count = defaultdict(int)
    user_count = defaultdict(int)
    user_hour_count = defaultdict(int)
    ff = open(feature_file, "wb")
    feature_map = {}
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
        smooth_user_count = user_hour_count[user_id + '-' + hour]
        ff.write(reader.read_row_data(row, "click") + "," + map_features([
            'pub_id-' + pub_id,
            'pub_domain-' + pub_domain,
            'pub_category-' + pub_category,
            'banner_pos-' + banner_pos,
            'device_model-' + device_model,
            'device_conn_type-' + device_conn_type,
            'C14-' + C14,
            'C17-' + C17,
            'C20-' + C20,
            'C21-' + C21,
            'hour-' + hour[-2:],
            device_ip_count[device_ip] > 1000 and 'device_ip-' + device_ip or 'device_ip-less-' + str(device_ip_count[device_ip]),
            device_id_count[device_id] > 1000 and 'device_id-' + device_id or 'device_id-less-' + str(device_id_count[device_id]),
            smooth_user_count > 30 and 'smooth_user_hour_count-0' or 'smooth_user_hour_count-' + str(smooth_user_count),
            str(user_count[user_id])
        ], feature_map) + "\r\n")
        utility.counting_line(reader.index)
    ff.close()
    with open(map_file, "wb") as f:
        for k, v in feature_map.items():
            f.write("{0}\t{1}\t{2}\r\n".format(k, v[0], v[1]))

if __name__ == "__main__":
    import os
    root = os.path.dirname(__file__)
    convert_feature(os.path.join(root, "../data/t.csv"), os.path.join(root, "../data/train_features.txt"), os.path.join(root, "../data/feature_map.txt"))