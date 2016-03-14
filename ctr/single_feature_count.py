#count single dimension features
from __future__ import absolute_import
from ctr.comon import utility
from collections import defaultdict
import pandas as pd


def count_featurs(train_file, output_path):
    lineno = 0
    chunksize = 100000

    reader = pd.read_csv(train_file, chunksize=chunksize)
    featurs_maps = {}
    for chunk in reader:
        lineno += chunksize
        utility.counting_line(lineno)
        # skip id, click, hour
        if len(featurs_maps) == 0:
            for column in chunk.columns[3:]:
                featurs_maps[column] = defaultdict(int)
        for column in chunk.columns[3:]:
            t = chunk[column].value_counts()
            for count, feature in enumerate(t.axes):
                featurs_maps[column][str(feature)] += t.values[count]


if __name__ == "__main__":
    count_featurs("../data/t.csv", "../data")