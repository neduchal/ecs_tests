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
from my_net import my_net
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


predictions_to_return = 9
model =  my_net(weights_path='/storage/plzen1/home/neduchal/projekty/env_detection_edges/src/net_edges_weights.h5')
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.Canny(image, 100, 200)
    image = cv2.resize(image, (320, 240))
    image = image.reshape(1, 240, 320, 1)   
    start = time.time()
    times.append((sensors_row[0] - first_timestamp)/(10**9))
    preds = np.argmax(model.predict(image)[0])
    last_pr = preds
    print(preds)
    predicts.append(preds)
    #vote += learning_rate * (-1 + predicts[-1]*2)
    #vote = max(min(vote, 1), -1)
    #voting_predicts.append(vote)
    duration.append(time.time() - start)
    processed += 1

print("\n")
print(np.mean(duration), np.sum(duration), processed)

plt.figure(1, figsize=(10, 4))
plt.plot(times, predicts)
plt.xlabel("time [s]")
plt.ylabel("predicts value")
plt.savefig(os.path.join(path, "mn_predicts.pdf"))

"""
plt.figure(2, figsize=(10, 4))
plt.plot(times, voting_predicts)
plt.xlabel("time [s]")
plt.ylabel("vote value")
plt.savefig(os.path.join(path, "mn_voting_predicts_centrist.pdf"))


plt.figure(3, figsize=(10, 4))
plt.plot(times, np.array(voting_predicts) >= 0)
plt.xlabel("time [s]")
plt.ylabel("threshold vote value]")
plt.savefig(os.path.join(path, "mn_threshold_voting_predicts_centrist.pdf"))
"""