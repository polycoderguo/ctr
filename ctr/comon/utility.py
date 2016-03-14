def counting_line(lineno, report_interval=1000000):
    if lineno % report_interval == 0 and lineno > 0:
        print "process line {0}......".format(lineno)
    return lineno + 1
