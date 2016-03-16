#count single dimension features
from __future__ import absolute_import
from ctr.comon import utility
from collections import defaultdict
import pandas as pd
import os


def count_featurs(train_file, output_path):
    lineno = 0
    chunksize = 100000

    reader = pd.read_csv(train_file, chunksize=chunksize)
    featurs_maps = {}
    for chunk in reader:
        lineno += chunksize
        utility.counting_line(lineno)
        # skip id, hour
        columns = chunk.columns[1:2] | chunk.columns[3:]
        if len(featurs_maps) == 0:
            for column in columns:
                featurs_maps[column] = defaultdict(int)
        for column in columns:
            t = chunk[column].value_counts()
            for count, feature in enumerate(t.index.values):
                featurs_maps[column][feature] += int(t.values[count])
    for feature_type, v in featurs_maps.items():
        with open(os.path.join(output_path, "{0}.txt").format(feature_type), "wb") as f:
            t = v.items()
            t.sort(key=lambda x: x[-1], reverse=True)
            for feature, count in t:
                f.write("{0}\t{1}\r\n".format(feature, count))


if __name__ == "__main__":
    count_featurs("../data/t.csv", "../data")