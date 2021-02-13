import os
import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2
from joblib import load
from sklearn import svm
import time
import glob
import common
import centrist

cl = centrist.load()

path = "/home/neduchal/Dokumenty/Data/disertace/record2255"
sensors_data = open(os.path.join(path, "output.txt"), 'r').read().split('\n')

gt_ts = open(os.path.join(path, "gt.txt"), "r").read().split("\n")

c_gt = 0
n_gt = 0
error = []

vote = -1.0
learning_rate = 0.5
pr = 0
last_pr = 0
clf_fn = "./joblibs/svm_centrist_ms_64.joblib"
clf = load(clf_fn)

im_fns = sorted(glob.glob(os.path.join(path, "imgs/*.png")))
im_fns_len = len(im_fns)

duration = []
predicts = []
voting_predicts = []
processed = 0
max_indoor_distance = 500.0
last_distance = 0.0
first_timestamp = float(sensors_data[0].split(",")[0])
times = []
temperatures = []
var_temp = np.double(0.0050)
times_nonv = []
times_visual = []

for i, im_fn in enumerate(im_fns):
    common.printProgressBar(i+1, im_fns_len, length=70, printEnd="\r")
    sensors_row = common.read_sensor_data(sensors_data, i)
    if sensors_row == None:
        continue
    #distance = sensors_row[4]
    if n_gt < len(gt_ts) and os.path.basename(im_fn) == gt_ts[n_gt].split(",")[0]:
        c_gt =  int(gt_ts[n_gt].split(",")[1])
        n_gt += 1       
    temperatures.append(sensors_row[1])
    start_nonv = time.time()    
    if len(temperatures) > 40:        
        temperatures.pop(0)
    if len(temperatures) < 40:
        continue    
    decision = common.temperature_decision_dirat(temperatures, 2)
    times_nonv.append(time.time() - start_nonv)
    """
    if decision:
        voting_predicts.append(vote)
        predicts.append(last_pr)
        error.append(abs(last_pr - c_gt))
        times.append((sensors_row[0] - first_timestamp)/(10**9))
        continue
    """
    im = cv2.imread(im_fn, 0)
    im = cv2.resize(im, (128, 128))
    times.append((sensors_row[0] - first_timestamp)/(10**9))
    start = time.time()    
    desc_vector = common.centrist_multiscale_desc(cl, im, bins=64)
    pr = clf.predict(desc_vector.reshape(1, -1))[0]
    times_visual.append(time.time() - start)
    error.append(abs(pr - c_gt))
    last_pr = pr
    predicts.append(pr)
    vote += learning_rate * (-1 + predicts[-1]*2)
    vote = max(min(vote, 1), -1)
    voting_predicts.append(vote)
    duration.append(time.time() - start)
    processed += 1

time_nonv = sum(times_nonv)
times_all = []
times_all.append(time_nonv)

for i in range(len(times_visual)):
    times_all.append(time_nonv + sum(times_visual[0:i]))

f = open(os.path.join(path, "speed_temperature.txt"), 'w')

f.write(str(len(times_nonv)) + "\n")
for item in times_nonv:
    f.write(str(item) + ",")
f.write("\n")
f.write(str(len(times_visual)) + "\n")
for item in times_visual:
    f.write(str(item) + ",")
f.write("\n")
f.close()

"""
plt.figure(1, figsize=(10, 4))
plt.plot(times_all)
plt.xlabel("number of processed images")
plt.ylabel("duration [s]")
plt.savefig(os.path.join(path, "speed_temperature_classic.pdf"))
"""

print(sum(error)/im_fns_len, sum(error), im_fns_len, np.mean(times_nonv), sum(times_nonv), np.mean(times_visual), sum(times_visual), np.mean(times_visual+times_nonv), sum(times_visual+times_nonv))


