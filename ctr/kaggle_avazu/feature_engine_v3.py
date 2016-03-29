from __future__ import absolute_import
from ctr.common import utility
from collections import defaultdict
import os


def is_app(row):
    return True if row.get("site_id") == '85f751fd' else False


def has_id_info(row):
    return False if row.get("device_id") == 'a99f214a' else True


def get_user_id(row):
    if has_id_info(row):
        user_id = 'id-' + row.get("device_id")
    else:
        user_id = 'ip-' + row.get("device_ip") + '-' + row.get("device_model")
    return user_id

device_ip_count = defaultdict(int)
device_id_count = defaultdict(int)
user_count = defaultdict(int)
user_hour_count = defaultdict(int)
history = defaultdict(lambda: {'history': '', 'buffer': '', 'prev_hour': ''})


def scan(train_file_name):
    reader = utility.CSVReader(train_file_name)
    for count, row in enumerate(reader):
        device_id = row.get("device_id")
        device_id_count[device_id] += 1
        device_ip = row.get("device_ip")
        device_ip_count[device_ip] += 1
        user_id = get_user_id(row)
        user_count[user_id] += 1
        hour = row.get("hour")
        user_hour_count[user_id + '-' + hour] += 1
        utility.progress(count)
    reader.close()


def convert_feature(train_file_name, feature_file_name, map_file_name, shared_map_file=None, is_train=True, submit=False):
    reader = utility.CSVReader(train_file_name)

    t = os.path.split(feature_file_name)
    app_feature_file_name = os.path.join(t[0], 'app_' + t[1])
    site_feature_file_name = os.path.join(t[0], 'site_' + t[1])

    if shared_map_file:
        feature_map = utility.FeatureMap.load(shared_map_file)
    else:
        feature_map = utility.FeatureMap()

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
        device_ip = row.get("device_ip")
        user_id = get_user_id(row)
        if has_id_info(row):
            if history[user_id]['prev_hour'] != row.get('hour'):
                history[user_id]['history'] = (history[user_id]['history'] + history[user_id]['buffer'])[-4:]
                history[user_id]['buffer'] = ''
                history[user_id]['prev_hour'] = row.get('hour')
            else:
                pass
            user_click_history = history[user_id]['history']
            if len(user_click_history) > 1:
                pass
            if is_train:
                history[user_id]['buffer'] += row.get('click')
        else:
            user_click_history = ''
        smooth_user_count = user_hour_count[user_id + '-' + hour]
        if app_row:
            ff = ff_app
        else:
            ff = ff_site
        if submit:
            append_info = "{0},{1}".format(row.get("id"), app_row and 1 or 0)
        else:
            append_info = row.get("click")
        ff.write(
            append_info + "," + feature_map.map_features([
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
            user_count[user_id] > 30 and 'user_click_history-' + str(user_count[user_id]) or 'user_click_history-' + str(user_count[user_id]) + '-' + user_click_history
        ], seq=',') + "\r\n")
        utility.progress(count)
    ff_app.close()
    ff_site.close()
    feature_map.save(map_file_name)

if __name__ == "__main__":
    print "Scan......"
    scan(utility.get_date_file_path("t.csv"))
    scan(utility.get_date_file_path("v.csv"))
    print "Train features....."
    convert_feature(utility.get_date_file_path("t.csv"), utility.get_date_file_path("train_features_v3.csv"), utility.get_date_file_path("feature_map_v3.json"))
    print "Test features....."
    convert_feature(utility.get_date_file_path("v.csv"), utility.get_date_file_path("test_features_v3.csv"),
                    utility.get_date_file_path("feature_map_v3.json"),
                    shared_map_file=utility.get_date_file_path("feature_map_v3.json"), is_train=False)