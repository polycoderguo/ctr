from __future__ import absolute_import

from ctr.comon import utility


def split_train_validation_data_set(raw_train_file, train_file, validation_file, test_samples_rate):
    print "count file lines"
    with open(raw_train_file, "rb") as f:
        for count, line in enumerate(f):
            utility.counting_line(count)
    print "total lines: {0}".format(count)
    train_samples_count = count - int(count * test_samples_rate)
    print "generate data set"
    with open(raw_train_file, "rb") as f:
        head = f.readline()
        with open(train_file, "wb") as tf:
            tf.write(head)
            with open(validation_file, "wb") as vf:
                for count, line in enumerate(f):
                    utility.counting_line(count + 1)
                    if count < train_samples_count:
                        tf.write(line)
                    elif count == train_samples_count:
                        vf.write(head)
                    else:
                        vf.write(line)
    print "finished"


if __name__ == "__main__":
    split_train_validation_data_set("../data/train.csv", "../data/t.csv", "../data/v.csv", 0.1)
