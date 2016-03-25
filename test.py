import gnumpy as gpu
import numpy as np
import time
import cudamat as cm
import math

t = time.time()
for i in xrange(100000):
    math.log(0.12)
print time.time() - t

t = time.time()
for i in xrange(100000):
    np.log(0.12)
print time.time() - t