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

path = "/media/neduchal/data/data/disertace/record0949"
sensors_data = open(os.path.join(path, "output.txt"), 'r').read().split('\n')

vote = -1.0
learning_rate = 0.5
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
temperatures = []
var_temp = np.double(0.0050)

for i, im_fn in enumerate(im_fns):
    common.printProgressBar(i+1, im_fns_len, length=70, printEnd="\r")
    sensors_row = common.read_sensor_data(sensors_data, i)
    if sensors_row == None:
        continue
    #distance = sensors_row[4]
    temperatures.append(sensors_row[1])
    if len(temperatures) > 40:
        temperatures.pop(0)
    if len(temperatures) < 40:
        continue
    if common.temperature_decision_dirat(temperatures, 2):
        voting_predicts.append(vote)
        predicts.append(pr)
        times.append((sensors_row[0] - first_timestamp)/(10**9))
        continue
    im = cv2.imread(im_fn, 0)
    im = cv2.resize(im, (128, 128))
    start = time.time()
    times.append((sensors_row[0] - first_timestamp)/(10**9))
    desc_vector = common.centrist_multiscale_desc(cl, im, bins=64)
    pr = clf.predict(desc_vector.reshape(1, -1))[0]
    last_pr = pr
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
plt.savefig(os.path.join(path, "temp_predicts_centrist.pdf"))


plt.figure(2, figsize=(10, 4))
plt.plot(times, voting_predicts)
plt.xlabel("time [s]")
plt.ylabel("vote value")
plt.savefig(os.path.join(path, "temp_voting_predicts_centrist.pdf"))


plt.figure(3, figsize=(10, 4))
plt.plot(times, np.array(voting_predicts) >= 0)
plt.xlabel("time [s]")
plt.ylabel("threshold vote value]")
plt.savefig(os.path.join(path, "temp_threshold_voting_predicts_centrist.pdf"))
