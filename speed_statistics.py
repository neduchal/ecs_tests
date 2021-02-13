import glob
import os
import numpy as np
from matplotlib import pyplot as plt

directory = "/home/neduchal/Dokumenty/Data/disertace/record2255"

fns = glob.glob(os.path.join(directory, "speed_distance_*.txt"))

print(len(fns))
plt.figure()
for i, fn in enumerate(fns):
    data = open(fn, "r").read().split("\n")
    data_nonv = np.array(data[1].split(",")[:-1], dtype="f")
    data_visual = np.array(data[3].split(",")[:-1], dtype="f")
    data1 = np.sum(data_nonv[0:30]) + data_visual[0]
    data5 = np.sum(data_nonv[0:30]) + np.sum(data_visual[0:5])
    data10 = np.sum(data_nonv[0:30]) + np.sum(data_visual[0:10])
    data30 = np.sum(data_nonv[0:30]) + np.sum(data_visual[0:30])
    plt.plot([0,1,5,10,30], [ np.sum(data_nonv[0:30]), data1, data5, data10, data30])
    print("%s avg without: %f, avg_with: %f, 1: %f  5: %f 10: %f 30: %f \n" % (os.path.basename(fn), np.mean(data_nonv), np.mean(data_nonv+data_visual), data1, data5, data10, data30) )

plt.show()    