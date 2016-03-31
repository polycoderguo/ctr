from __future__ import absolute_import
import json
import datetime
import os
from collections import defaultdict
import time
import math
import hashlib


def wise_mk_dir(path):
    if path == "":
        return
    if os.path.exists(path):
        return
    p, c = os.path.split(path)
    if not os.path.exists(p):
        wise_mk_dir(p)
    os.mkdir(path)


def wise_mk_dir_for_file(filepath):
    p = os.path.dirname(filepath)
    wise_mk_dir(p)


def format_rate(a, b):
    if b == 0:
        return "N/A"
    return "{0:%}".format(float(a) / float(b))


class ValidateHelper(object):
    def __init__(self, report_interval=1000000):
        self.total = 0
        self.clicked = 0
        self.predict_clicked_correct = 0
        self.predict_clicked_wrong = 0
        self.predict_un_clicked_correct = 0
        self.predict_un_clicked_wrong = 0
        self.loss = 0
        self.avg_loss = 0
        self.report_interval = report_interval
        self.last_time = time.time()

    def update(self, p, clicked, p_threshold):
        self.total += 1
        if clicked:
            self.clicked += 1
        if p > p_threshold:
            if clicked:
                self.predict_clicked_correct += 1
            else:
                self.predict_clicked_wrong += 1
        else:
            if clicked:
                self.predict_un_clicked_wrong += 1
            else:
                self.predict_un_clicked_correct += 1
        #self.loss += clicked * np.log(p) + (1 - clicked) * np.log(1 - p)
        epsilon = 1e-15
        p = min(1-epsilon, max(epsilon, p))
        if clicked:
            self.loss += math.log(p)
        else:
            self.loss += math.log(1-p)
        avg = self.clicked / float(self.total)
        avg = min(1-epsilon, max(epsilon, avg))
        self.avg_loss += clicked * math.log(avg) + (1 - clicked) * math.log(1 - avg)
        if self.report_interval > 0 and self.total % self.report_interval == 0:
            self.out_put()

    def get_log_loss(self):
        if self.loss != 0:
            logloss = (-1.0 / float(self.total)) * self.loss
        else:
            logloss = 0
        return logloss

    def out_put(self):
        logloss = self.get_log_loss()
        if self.avg_loss != 0:
            avg_logloss = (-1.0 / float(self.total)) * self.avg_loss
        else:
            avg_logloss = 0
        print "total = {0}, clicked = {1}, click_rate = {2}, predict_clicked_correct = {3}, predict_clicked_wrong = {4}" \
              ", predict_un_clicked_correct = {5}, predict_un_clicked_wrong = {6}, precision = {7}, recall = {8}, " \
              "logloss = {9}, avgloss = {10}, time = {11}"\
            .format(self.total, self.clicked, format_rate(self.clicked, self.total), self.predict_clicked_correct
                    , self.predict_clicked_wrong, self.predict_un_clicked_correct, self.predict_un_clicked_wrong
                    , format_rate(self.predict_clicked_correct, self.predict_clicked_correct + self.predict_clicked_wrong)
                    , format_rate(self.predict_clicked_correct, self.clicked), logloss, avg_logloss
                    , time.strftime("%Y-%m-%d %H:%M:%S"))


def progress(count, report_interval=1000000):
    if count % report_interval == 0:
        print "process {0}......{1}".format(count, time.strftime("%Y-%m-%d %H:%M:%S"))
    return count + 1


def count_file_lines(fname):
    count = 0
    with open(fname, "rb") as f:
        for count, _ in enumerate(f):
            pass
    return count + 1


def split_train_test_data_set(raw_file, train_file, test_file, test_samples_rate=0.1, headers=True):
    print "count file lines......"
    file_lines = count_file_lines(raw_file)
    if headers:
        count = file_lines - 1
    else:
        count = file_lines
    train_samples_count = count - int(count * test_samples_rate)
    print "total samples = {0}, train_samples_count = {1}, writing files......".format(count, train_samples_count)
    with open(raw_file, "rb") as f:
        if headers:
            head = f.readline()
        with open(train_file, "wb") as tf:
            if headers:
                tf.write(head)
            with open(test_file, "wb") as vf:
                for count, line in enumerate(f):
                    if count < train_samples_count:
                        tf.write(line)
                    elif count == train_samples_count and headers:
                        vf.write(head)
                    else:
                        vf.write(line)
                    progress(count)


class CSVReader(object):
    class CSVRow(object):
        def __init__(self,  data, columns_map):
            self.data = data
            self.columns_map = columns_map

        def get(self, row_name):
            return self.data[self.columns_map[row_name]]

    def __init__(self, fname, from_line=0, end_line=0, seq=","):
        self.f = open(fname, "rb")
        self.seq = seq
        t = self.f.readline().strip().split(self.seq)
        self.columns_map = {}
        for i in range(len(t)):
            self.columns_map[t[i]] = i
        self.index = 0
        self.end_line = end_line
        if from_line > 0:
            self.skip_rows(from_line)

    def __iter__(self):
        return self

    def next(self):
        if 0 < self.end_line <= self.index:
            raise StopIteration()
        try:
            t = self.f.readline().strip().split(self.seq)
            assert len(t) == len(self.columns_map)
            self.index += 1
            return CSVReader.CSVRow(t, self.columns_map)
        except:
            raise StopIteration()

    def skip_rows(self, count):
        for i in xrange(count):
            self.f.readline()

    def close(self):
        self.f.close()


class FeatureMap(object):
    def __init__(self):
        self.index = -1
        self.feature_index_map = {}
        self.features = []
        self.total_fields = 0

    def get_feature_id(self, str_feature):
        try:
            return self.feature_index_map[str_feature]
        except:
            self.index += 1
            self.feature_index_map[str_feature]=str(self.index)
            self.features.append(str_feature)
            return self.feature_index_map[str_feature]

    def map_features(self, features, seq=","):
        t = []
        for feature in features:
            try:
                t.append(self.get_feature_id(feature))
            except Exception as e:
                pass
        self.total_fields = max(self.total_fields, len(t))
        return seq.join(t)

    def max_feature(self):
        return self.index + 1

    def max_fields(self):
        return self.total_fields

    def save(self, feature_data_filename):
        with open(feature_data_filename, "wb") as f:
            json.dump(self.__dict__, f)

    @staticmethod
    def load(feature_data_filename):
        with open(feature_data_filename, "rb") as f:
            t = json.load(f)
            feature_map = FeatureMap()
            feature_map.__dict__ = t
            return feature_map

    def feature_index_2_str(self, feature_index):
        return self.features[feature_index]

NR_BINS = 1000000


def hashstr(input):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16) % (NR_BINS-1)+1)


class HashFeatureMap(object):
    def __init__(self):
        self.feature_index_map = {}
        self.total_fields = 0

    def get_feature_id(self, str_feature):
        feature = hashstr(str_feature)
        self.feature_index_map[feature] = str_feature
        return feature

    def map_features(self, features, seq=","):
        t = []
        for feature in features:
            try:
                t.append(self.get_feature_id(feature))
            except Exception as e:
                pass
        self.total_fields = max(self.total_fields, len(t))
        return seq.join(t)

    def max_feature(self):
        return NR_BINS + 1

    def max_fields(self):
        return self.total_fields

    def save(self, feature_data_filename):
        with open(feature_data_filename, "wb") as f:
            json.dump(self.__dict__, f)

    @staticmethod
    def load(feature_data_filename):
        with open(feature_data_filename, "rb") as f:
            t = json.load(f)
            feature_map = HashFeatureMap()
            feature_map.__dict__ = t
            return feature_map

    def feature_index_2_str(self, feature_index):
        return self.feature_index_map[feature_index]


class FeatureStream(object):
    def __init__(self, feature_map_filename, feature_data_filename, seq=",", click_at=0):
        self.feature_map_filename = feature_map_filename
        self.feature_data_filename = feature_data_filename
        self.seq = seq
        self.click_at = click_at
        self.reset()

    def reset(self):
        self.feature_map = HashFeatureMap.load(self.feature_map_filename)
        self.f = open(self.feature_data_filename, "rb")

    def __iter__(self):
        return self

    def feature_count(self):
        return self.feature_map.max_feature()

    def field_count(self):
        return self.feature_map.max_fields()

    def next(self):
        try:
            t = self.f.readline().strip()
            assert len(t) > 0
            t = t.split(self.seq)
            return t[:self.click_at], int(t[self.click_at]), [int(x) for x in t[self.click_at + 1:]]
        except:
            raise StopIteration()


DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data")


def get_date_file_path(filename):
    return os.path.join(DEFAULT_DATA_PATH, filename)


def count_csv_values_by_field(filename, output_dirname, seq=","):
    fields_values_map = defaultdict(dict)
    with open(filename, "rb") as f:
        head = f.readline().strip().split(seq)
        for count, line in enumerate(f):
            t = line.strip().split(seq)
            if len(line) == 0:
                continue
            for index, item in enumerate(t):
                try:
                    fields_values_map[head[index]][item] += 1
                except:
                    fields_values_map[head[index]][item] = 1
            progress(count)
    for field, v in fields_values_map.iteritems():
        file_path = os.path.join(output_dirname, field)
        wise_mk_dir_for_file(file_path)
        with open(file_path, "wb") as f:
            t = v.items()
            t.sort(key=lambda x: x[-1])
            t.reverse()
            for item, count in t:
                f.write("{0}\t{1}\r\n".format(item, count))


class Timer(object):
    def __init__(self):
        self.tm = 0
        self.count = 0
        self.total_time = 0.0

    def reset(self):
        self.tm = 0
        self.count = 0
        self.total_time = 0.0

    def update_begin(self):
        self.tm = datetime.datetime.utcnow()

    def update_end(self):
        self.total_time += (datetime.datetime.utcnow() - self.tm).total_seconds()
        self.count += 1

    def avg_cost_time(self):
        return self.count > 0 and self.total_time / float(self.count) or 0

    def total_cost_time(self):
        return self.total_time

def sigmoid(x):
  return 1.0 / (1.0 + math.exp(-x))


def sign(x):
    if x < 0:
        return -1
    elif x == 0:
        return 0
    else:
        return 1