import json
import time
import os
import numpy as np


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
        self.loss += clicked * np.log(p) + (1 - clicked) * np.log(1 - p)
        avg = self.clicked / float(self.total)
        if avg > 0:
            self.avg_loss += clicked * np.log(avg) + (1 - clicked) * np.log(1 - avg)
        if self.total % self.report_interval == 0:
            self.out_put()

    def out_put(self):
        logloss = (-1.0 / float(self.total)) * self.loss
        avg_logloss = (-1.0 / float(self.total)) * self.avg_loss
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
        return seq.join(t)

    def max_feature(self):
        return self.index + 1

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


class FeatureStream(object):
    def __init__(self, feature_map_filename, feature_data_filename, seq=","):
        self.feature_map = FeatureMap.load(feature_map_filename)
        self.f = open(feature_data_filename, "rb")
        self.seq = seq

    def __iter__(self):
        return self

    def feature_count(self):
        return self.feature_map.max_feature()

    def next(self):
        try:
            t = self.f.readline().strip()
            assert len(t) > 0
            t = t.split(self.seq)
            return int(t[0]), (int(x) for x in t[1:])
        except:
            raise StopIteration()

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data")


def get_date_file_path(filename):
    return os.path.join(DEFAULT_DATA_PATH, filename)