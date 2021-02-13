import os
import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2
from joblib import load
#from sklearn import svm
import time
import glob
import common
import centrist
from vgg16_places_365 import VGG16_Places365
from PIL import Image

#cl = centrist.load()

path = "/home/neduchal/Dokumenty/Data/disertace/record2255"
sensors_data = open(os.path.join(path, "output.txt"), 'r').read().split('\n')
file_name = 'categories_places365_io.txt'

#gt_ts = open(os.path.join(path, "gt.txt"), "r").read().split("\n")
c_gt = 0
n_gt = 0
error = []

vote = -1.0
learning_rate = 0.3
pr = 0

# clf_fn = "./joblibs/svm_centrist_ms_64.joblib"
#clf = load(clf_fn)

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
classes = list()
cl_type = list()
temperatures = []
times_nonv = []
times_visual = []

with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
        cl_type.append(line.strip().split(' ')[2])


predictions_to_return = 9
model = VGG16_Places365(weights='places')
for i, im_fn in enumerate(im_fns):
    common.printProgressBar(i+1, im_fns_len, length=70, printEnd="\r")
    sensors_row = common.read_sensor_data(sensors_data, i)
    if sensors_row == None:
        continue
    #if n_gt < len(gt_ts) and os.path.basename(im_fn) == gt_ts[n_gt].split(",")[0]:
    #    c_gt =  int(gt_ts[n_gt].split(",")[1])
    #    n_gt += 1       
    temperatures.append(sensors_row[1])
    start_nonv = time.time()    
    if len(temperatures) > 40:
        temperatures.pop(0)
    if len(temperatures) < 40:
        continue
    decision = common.temperature_decision_dirat(temperatures, 1.2)
    times_nonv.append(time.time() - start_nonv)
    #voting_predicts.append(vote)
    #predicts.append(pr)
    #error.append(abs(pr - c_gt))        
    #times.append((sensors_row[0] - first_timestamp)/(10**9))
    #continue
    image = Image.open(im_fn)
    image = np.array(image, dtype=np.uint8)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, 0)
    start_vis = time.time()    
    start = time.time()
    times.append((sensors_row[0] - first_timestamp)/(10**9))
    preds = model.predict(image)[0]
    times_visual.append(time.time() - start_vis)
    top_preds = np.argsort(preds)[::-1][0:predictions_to_return]
    # output the prediction
    stats = [0, 0, 0]
    stats2 = [0, 0, 0]
    for i in range(0, predictions_to_return):
        #print(classes[top_preds[i]] + ", ", end='')
        if cl_type[top_preds[i]] == "I":
            stats2[0] += preds[top_preds[i]]
            stats[0] += 1
        elif cl_type[top_preds[i]] == "O":
            stats2[1] += preds[top_preds[i]]
            stats[1] += 1
        else:
            stats2[2] += preds[top_preds[i]]
            stats[2] += 1
    if stats2[0] > stats2[1]:
        pr = 0
    else:
        pr = 1
    last_pr = pr
    error.append(abs(pr - c_gt))    
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

print(len(times_nonv), len(times_visual))

f = open(os.path.join(path, "speed_temperature_vgg16_gpu.txt"), 'w')

f.write(str(len(times_nonv)) + "\n")
for item in times_nonv:
    f.write(str(item) + ",")
f.write("\n")
f.write(str(len(times_visual)) + "\n")
for item in times_visual:
    f.write(str(item) + ",")
f.write("\n")
f.close()

plt.figure(1, figsize=(10, 4))
plt.plot(times_all)
plt.xlabel("number of processed images")
plt.ylabel("duration [s]")
plt.savefig(os.path.join(path, "speed_temperature_vgg16_gpu.pdf"))
