import hashlib
import time

NR_BINS = 1000000


def hashstr(input):
    return str(int(hashlib.md5(input.encode('utf8')).hexdigest(), 16) % (NR_BINS-1)+1)


def counting_line(lineno, report_interval=1000000):
    if lineno % report_interval == 0:
        print "process line {0}......{1}".format(lineno, time.strftime("%Y-%m-%d %H:%M:%S"))
    return lineno + 1


def is_app(site_id):
    return True if site_id == '85f751fd' else False


def has_id_info(device_id):
    return False if device_id == 'a99f214a' else True


class CSVReader(object):
    def __init__(self, fname):
        self.f = open(fname, "rb")
        t = self.f.readline().strip().split(",")
        self.columns_map = {}
        for i in range(len(t)):
            self.columns_map[t[i]] = i
        self.index = 0

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