from __future__ import absolute_import
from ctr.common import utility
from collections import defaultdict
import os


def is_app(row):
    return True if row.get("site_id") == '85f751fd' else False


def has_id_info(row):
    return False if row.get("device_id") == 'a99f214a' else True


def convert_feature(train_file_name, feature_file_name, map_file_name, shared_app_map_file=None, shared_site_map_file=None):
    reader = utility.CSVReader(train_file_name)
    device_ip_count = defaultdict(int)
    device_id_count = defaultdict(int)
    user_count = defaultdict(int)
    user_hour_count = defaultdict(int)

    t = os.path.split(feature_file_name)
    app_feature_file_name = os.path.join(t[0], 'app_' + t[1])
    site_feature_file_name = os.path.join(t[0], 'site_' + t[1])

    t = os.path.split(map_file_name)
    app_map_file = os.path.join(t[0], 'app_' + t[1])
    site_map_file = os.path.join(t[0], 'site_' + t[1])

    if shared_app_map_file:
        app_feature_map = utility.DummyFeatureMap.load(shared_app_map_file)
    else:
        app_feature_map = utility.DummyFeatureMap()
    if shared_site_map_file:
        site_feature_map = utility.DummyFeatureMap.load(shared_site_map_file)
    else:
        site_feature_map = utility.DummyFeatureMap()

    ff_app = open(app_feature_file_name, "wb")
    ff_site = open(site_feature_file_name, "wb")

    for count, row in enumerate(reader):
        app_row = False
        if is_app(row):
            app_row = True
            pub_id = row.get("app_id")
            pub_domain = row.get("app_domain")
            pub_category = row.get("app_category")
        else:
            pub_id = row.get("site_id")
            pub_domain = row.get("site_domain")
            pub_category = row.get("site_category")
        banner_pos = row.get("banner_pos")
        device_model = row.get("device_model")
        device_conn_type = row.get("device_conn_type")
        C14, C17, C20, C21 = row.get("C14"), row.get("C17"), row.get("C20"), row.get("C21")
        hour = row.get("hour")
        device_id = row.get("device_id")
        device_id_count[device_id] += 1
        device_ip = row.get("device_ip")
        device_ip_count[device_ip] += 1
        if has_id_info(row):
            user_id = device_id
        else:
            user_id = device_ip + device_model
        user_count[user_id] += 1
        user_hour_count[user_id + '-' + hour] += 1
        smooth_user_count = user_hour_count[user_id + '-' + hour]
        if app_row:
            ff = ff_app
            feature_map = app_feature_map
        else:
            ff = ff_site
            feature_map = site_feature_map
        ff.write(row.get("click") + "," + feature_map.map_features([
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
            'user_count-' + str(user_count[user_id])
        ]) + "\r\n")
        utility.progress(count)
    ff_app.close()
    ff_site.close()
    app_feature_map.save(app_map_file)
    site_feature_map.save(site_map_file)

if __name__ == "__main__":
    print "Train features....."
    convert_feature(utility.get_date_file_path("t.csv"), utility.get_date_file_path("train_features.csv"), utility.get_date_file_path("feature_map.json"))
    print "Test features....."
    convert_feature(utility.get_date_file_path("v.csv"), utility.get_date_file_path("test_features.csv"),
                    utility.get_date_file_path("feature_map.json"), shared_app_map_file=utility.get_date_file_path("app_feature_map.json"), shared_site_map_file=utility.get_date_file_path("site_feature_map.json"))