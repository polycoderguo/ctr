from __future__ import absolute_import
from ctr.common import utility
from ctr.common.utility import CSVReader
from collections import defaultdict
import multiprocessing
import os


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


def convert_feature(train_file, feature_file, map_file, startlines=0, maxlines=-1):
    reader = CSVReader(train_file)
    reader.skip_rows(startlines)
    device_ip_count = defaultdict(int)
    device_id_count = defaultdict(int)
    user_count = defaultdict(int)
    user_hour_count = defaultdict(int)
    t = os.path.split(feature_file)
    feature_file_app = os.path.join(t[0], 'app_' + t[1])
    feature_file_site = os.path.join(t[0], 'site_' + t[1])
    t = os.path.split(map_file)
    map_file_app = os.path.join(t[0], 'app_' + t[1])
    map_file_site = os.path.join(t[0], 'site_' + t[1])

    ff_app = open(feature_file_app, "wb")
    ff_site = open(feature_file_site, "wb")
    feature_map_app = {}
    feature_map_site = {}
    count = 0
    while True:
        if count >= maxlines:
            break
        row = reader.read_row()
        count += 1
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
        if is_app:
            ff = ff_app
            feature_map = feature_map_app
        else:
            ff = ff_site
            feature_map = feature_map_site
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
            'user_count-' + str(user_count[user_id])
        ], feature_map) + "\r\n")
        utility.counting_line(reader.index)
    ff_app.close()
    ff_site.close()
    with open(map_file_app, "wb") as f:
        for k, v in feature_map_app.items():
            f.write("{0}\t{1}\t{2}\r\n".format(k, v[0], v[1]))
    with open(map_file_site, "wb") as f:
        for k, v in feature_map_site.items():
            f.write("{0}\t{1}\t{2}\r\n".format(k, v[0], v[1]))


if __name__ == "__main__":
    import os
    root = os.path.dirname(__file__)
    process = []
    train_file = os.path.join(root, "../data/v.csv")
    total_lines = utility.count_file_lines(train_file)
    print "total lines:", total_lines
    thread = 4
    for i in range(thread):
        print "start thread {0}, from lines {1}".format(i,total_lines/thread * i)
        p = multiprocessing.Process(target=convert_feature, args=(train_file, os.path.join(root, "../data/validate_features_{0}.txt".format(i)), os.path.join(root, "../data/feature_map_{0}.txt".format(i)), total_lines/thread * i, total_lines/thread))
        p.start()
        process.append(p)
    for p in process:
        p.join()
    print "merge train file"
    for t in ("app", "site"):
        with open(os.path.join(root, "../data/{0}_validate_features.txt").format(t), "wb") as wf:
            for i in range(thread):
                print "merge {0} train file {1}".format(t, i)
                with open(os.path.join(root, "../data/{0}_validate_features_{1}.txt".format(t, i)), "rb") as f:
                    for count, line in enumerate(f):
                        wf.write(line)
    if 0 > 1:

        for t in ("app", "site"):
            feature_map = {}
            with open(os.path.join(root, "../data/{0}_feature_map.txt").format(t), "wb") as wf:
                for i in range(thread):
                    print "merge {0} map file {1}".format(t, i)
                    with open(os.path.join(root, "../data/{0}_feature_map_{1}.txt".format(t, i)), "rb") as f:
                        for count, line in enumerate(f):
                            try:
                                items = line.strip().split("\t")
                                feature_map[items[0]][1] += items[2]
                            except:
                                feature_map[items[0]] = [items[1], int(items[2])]
                for k, v in feature_map.items():
                    wf.write("{0}\t{1}\t{2}\r\n".format(k, v[0], v[1]))