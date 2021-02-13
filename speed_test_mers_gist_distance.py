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
import leargist
from PIL import Image

cl = centrist.load()

path = "/home/neduchal/Dokumenty/Data/disertace/record2255"
sensors_data = open(os.path.join(path, "output.txt"), 'r').read().split('\n')
gt_ts = open(os.path.join(path, "gt.txt"), "r").read().split("\n")

c_gt = 0
n_gt = 0
error = []

vote = -1.0
learning_rate = 1.0
pr = 0

clf_fn = "./joblibs/svm_gist.joblib"
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
times_nonv = []
times_visual = []

for i, im_fn in enumerate(im_fns):
    common.printProgressBar(i+1, im_fns_len, length=70, printEnd="\r")
    sensors_row = common.read_sensor_data(sensors_data, i)
    if sensors_row == None:
        continue
    distance = sensors_row[4]
    if n_gt < len(gt_ts) and os.path.basename(im_fn) == gt_ts[n_gt].split(",")[0]:
        c_gt =  int(gt_ts[n_gt].split(",")[1])
        n_gt += 1   
    decision, dt = common.distance_decision(distance, last_distance, max_indoor_distance)
    times_nonv.append(dt)
    #voting_predicts.append(vote)
    #predicts.append(pr)
    #error.append(abs(pr - c_gt))
    #last_distance = distance
    times.append((sensors_row[0] - first_timestamp)/(10**9))
    last_distance = distance
    #im = cv2.imread(im_fn, 0)
    #im = common.normalized(cv2.imread(im_fn))
    im = Image.open(im_fn).resize((128,128))
    start = time.time()    
    desc_vector = leargist.color_gist(im)
    print(desc_vector)
    #desc_vector = np.array(leargist.color_gist(im).tolist())       
    #im = cv2.resize(im, (128, 128))

    #desc_vector = common.centrist_multiscale_desc(cl, im, bins=64)
    pr = clf.predict(desc_vector.reshape(1, -1))[0]
    print(pr)
    times_visual.append(time.time() - start)
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

time_nonv = sum(times_nonv)
times_all = []
times_all.append(time_nonv)

for i in range(len(times_visual)):
    times_all.append(time_nonv + sum(times_visual[0:i]))

print(len(times_nonv), len(times_visual))

f = open(os.path.join(path, "speed_distance.txt"), 'w')

f.write(str(len(times_nonv)) + "\n")
for item in times_nonv:
    f.write(str(item) + ",")

f.write("\n")
f.write(str(len(times_visual)) + "\n")
for item in times_visual:
    f.write(str(item) + ",")
f.write("\n")
f.close()

#plt.figure(1, figsize=(10, 4))
#plt.plot(times_all)
#plt.xlabel("number of processed images")
#plt.ylabel("duration [s]")
#plt.savefig(os.path.join(path, "speed_distance_classic.pdf"))

plt.figure(3, figsize=(10, 4))
plt.plot(times, error)
plt.xlabel("time [s]")
plt.ylabel("error value]")
plt.savefig(os.path.join(path, "gist_error_distance.pdf"))

print(sum(error)/im_fns_len, sum(error), im_fns_len, np.mean(times_nonv), sum(times_nonv), np.mean(times_visual), sum(times_visual), np.mean(times_visual+times_nonv), sum(times_visual+times_nonv))
