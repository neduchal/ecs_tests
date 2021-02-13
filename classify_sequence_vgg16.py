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

path = "/media/neduchal/data/data/disertace/record0949"
sensors_data = open(os.path.join(path, "output.txt"), 'r').read().split('\n')
file_name = 'categories_places365_io.txt'

vote = -1.0
learning_rate = 0.2
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

with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
        cl_type.append(line.strip().split(' ')[2])


predictions_to_return = 5
model = VGG16_Places365(weights='places')
for i, im_fn in enumerate(im_fns):
    common.printProgressBar(i+1, im_fns_len, length=70, printEnd="\r")
    sensors_row = common.read_sensor_data(sensors_data, i)
    if sensors_row == None:
        continue
    # distance = sensors_row[4]

    # if common.distance_decision(distance, last_distance, max_indoor_distance):
    #     voting_predicts.append(vote)
    #     predicts.append(pr)
    #     last_distance = distance
    #     times.append((sensors_row[0] - first_timestamp)/(10**9))
    #     continue
    # last_distance = distance
    image = Image.open(im_fn)
    image = np.array(image, dtype=np.uint8)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, 0)
    start = time.time()
    times.append((sensors_row[0] - first_timestamp)/(10**9))
    preds = model.predict(image)[0]
    top_preds = np.argsort(preds)[::-1][0:predictions_to_return]
    # output the prediction
    stats = [0, 0, 0]
    stats2 = [0, 0, 0]
    for i in range(0, predictions_to_return):
        print(classes[top_preds[i]] + ", ", end='')
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
plt.savefig(os.path.join(path, "vgg_predicts_centrist5.pdf"))


plt.figure(2, figsize=(10, 4))
plt.plot(times, voting_predicts)
plt.xlabel("time [s]")
plt.ylabel("vote value")
plt.savefig(os.path.join(path, "vgg_voting_predicts_centrist5.pdf"))


plt.figure(3, figsize=(10, 4))
plt.plot(times, np.array(voting_predicts) >= 0)
plt.xlabel("time [s]")
plt.ylabel("threshold vote value]")
plt.savefig(os.path.join(path, "vgg_threshold_voting_predicts_centrist5.pdf"))
