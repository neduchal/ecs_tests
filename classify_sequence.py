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

path = "/media/neduchal/data/data/disertace/record2"
sensors_data = open(os.path.join(path, "output.txt"), 'r').read().split('\n')

gt_ts = open(os.path.join(path, "gt.txt"), "r").read().split("\n")
c_gt = 0
n_gt = 0

error = []
vote = -1.0
learning_rate = 0.3
pr = 0

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

for i, im_fn in enumerate(im_fns):
    common.printProgressBar(i+1, im_fns_len, length=70, printEnd="\r")
    sensors_row = common.read_sensor_data(sensors_data, i)
    if sensors_row == None:
        continue
    if n_gt < len(gt_ts) and os.path.basename(im_fn) == gt_ts[n_gt].split(",")[0]:
        c_gt =  int(gt_ts[n_gt].split(",")[1])
        n_gt += 1    
    # distance = sensors_row[4]

    # if common.distance_decision(distance, last_distance, max_indoor_distance):
    #     voting_predicts.append(vote)
    #     predicts.append(pr)
    #     last_distance = distance
    #     times.append((sensors_row[0] - first_timestamp)/(10**9))
    #     continue
    # last_distance = distance
    im = cv2.imread(im_fn, 0)
    im = cv2.resize(im, (128, 128))
    start = time.time()
    times.append((sensors_row[0] - first_timestamp)/(10**9))
    desc_vector = common.centrist_multiscale_desc(cl, im, bins=64)
    pr = clf.predict(desc_vector.reshape(1, -1))[0]
    last_pr = pr
    error.append(abs(pr - c_gt))
    predicts.append(pr)
    vote += learning_rate * (-1 + predicts[-1]*2)
    vote = max(min(vote, 1), -1)
    voting_predicts.append(vote)
    duration.append(time.time() - start)
    processed += 1

print("\n")
print(np.mean(duration), np.sum(duration), processed)

plt.figure(1, figsize=(10, 4))
plt.plot(times, predicts)
plt.xlabel("time [s]")
plt.ylabel("predicts value")
plt.savefig(os.path.join(path, "predicts_centrist_no_sensor.pdf"))


plt.figure(2, figsize=(10, 4))
plt.plot(times, voting_predicts)
plt.xlabel("time [s]")
plt.ylabel("vote value")
plt.savefig(os.path.join(path, "voting_predicts_centrist_no_sensor.pdf"))


plt.figure(3, figsize=(10, 4))
plt.plot(times, np.array(voting_predicts) >= 0)
plt.xlabel("time [s]")
plt.ylabel("threshold vote value]")
plt.savefig(os.path.join(path, "threshold_voting_predicts_centrist_no_sensor.pdf"))

plt.figure(3, figsize=(10, 4))
plt.plot(times, error)
plt.xlabel("time [s]")
plt.ylabel("error value]")
plt.savefig(os.path.join(path, "error_no_sensor.pdf"))

print(sum(error)/im_fns_len)