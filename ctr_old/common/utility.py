import hashlib
import time
import numpy as np

NR_BINS = 1000000


def hashstr(input):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16) % (NR_BINS-1)+1)


def counting_line(lineno, report_interval=1000000):
    if lineno % report_interval == 0:
        print "process line {0}......{1}".format(lineno, time.strftime("%Y-%m-%d %H:%M:%S"))
    return lineno + 1


class LogLossCounter(object):
    def __init__(self):
        self.count = 0
        self.sum = 0

    def count_logloss(self, p, y, report_interval=10000):
        self.count += 1
        loss = y * np.log(p) + (1-y) * np.log(1 - p)
        self.sum += loss
        if self.count % report_interval == 0:
            self.output()

    def output(self):
        logloss = (-1.0 / float(self.count)) * self.sum
        print "process line {0}, logloss={1}......{2}".format(self.count, logloss, time.strftime("%Y-%m-%d %H:%M:%S"))


def is_app(site_id):
    return True if site_id == '85f751fd' else False


def has_id_info(device_id):
    return False if device_id == 'a99f214a' else True


def count_file_lines(fname):
    with open(fname, "rb") as f:
        for count, _ in enumerate(f):
            pass
    return count + 1


class CSVReader(object):
    def __init__(self, fname):
        self.f = open(fname, "rb")
        t = self.f.readline().strip().split(",")
        self.columns_map = {}
        for i in range(len(t)):
            self.columns_map[t[i]] = i
        self.index = 0

    def skip_rows(self, count):
        for i in xrange(count):
            self.f.readline()

    def read_row(self):
        try:
            t = self.f.readline().strip().split(",")
            assert len(t) == len(self.columns_map)
            self.index += 1
            return t
        except:
            return None

    def read_row_data(self, row, row_name):
        return row[self.columns_map[row_name]]

    def close(self):
        self.f.close()