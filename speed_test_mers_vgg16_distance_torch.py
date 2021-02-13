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
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
import pretrainedmodels
from PIL import Image

#cl = centrist.load()

classes = ('outdoor', 'indoor')
num_classes = len(classes)
batch_size = 8

#net = models.vgg16_bn(pretrained=False)
#num_ftrs = net.classifier[6].in_features
#net.classifier[6] = nn.Linear(num_ftrs,2)

#net = models.densenet161(pretrained=False)
#num_ftrs = net.classifier.in_features
#net.classifier = nn.Linear(num_ftrs, num_classes)

#net = models.resnet50(pretrained=False)
#num_ftrs = net.fc.in_features
#net.fc = nn.Linear(num_ftrs, num_classes)

#net = pretrainedmodels.inceptionresnetv2()
#net.avgpool_1a = nn.AvgPool2d(kernel_size=2, stride =2)
#num_ftrs = net.last_linear.in_features
#net.last_linear = nn.Linear(num_ftrs, num_classes)

#net = pretrainedmodels.inceptionv4()
#set_parameter_requires_grad(net, feature_extract)
#net.avg_pool = nn.AvgPool2d(kernel_size=2, stride =2)
#num_ftrs = net.last_linear.in_features
#net.last_linear = nn.Linear(num_ftrs, num_classes)


#net = models.wide_resnet50_2(pretrained=True)
#set_parameter_requires_grad(net, feature_extract)
#num_ftrs = net.fc.in_features
#net.fc = nn.Linear(num_ftrs, num_classes)

net = models.resnext50_32x4d(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, num_classes)

#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#net = pretrainedmodels.xception()
#num_ftrs = net.last_linear.in_features
#net.last_linear = nn.Linear(num_ftrs, num_classes)
#set_parameter_requires_grad(net, feature_extract)
#normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomResizedCrop(size=size, scale=(0.5,1.0), ratio=(1.0,1.0)), transforms.ToTensor(), normalize])

#net.load_state_dict(torch.load('models/VGG16_best.pth'))
#net.load_state_dict(torch.load('models/DenseNet_best.pth'))
#net.load_state_dict(torch.load('models/InceptionResNetv2_best.pth'))
#net.load_state_dict(torch.load('models/InceptionV4_best.pth'))
#net.load_state_dict(torch.load('models/Xception_best.pth'))
#net.load_state_dict(torch.load('models/ResNetWide_best.pth'))
net.load_state_dict(torch.load('models/ResNext_best.pth'))

net.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
net = net.to(device)

path = "/home/neduchal/Dokumenty/Data/disertace/record2255"
sensors_data = open(os.path.join(path, "output.txt"), 'r').read().split('\n')
file_name = 'categories_places365_io.txt'

gt_ts = open(os.path.join(path, "gt.txt"), "r").read().split("\n")
c_gt = 0
n_gt = 0
error = []

loader = transforms.Compose([ transforms.ToTensor()])
triggers = []
vote = -1.0
learning_rate = 1.0
pr = 0

# clf_fn = "./joblibs/svm_centrist_ms_64.joblib"
#clf = load(clf_fn)
last_pr = 0
im_fns = sorted(glob.glob(os.path.join(path, "imgs/*.png")))
im_fns_len = len(im_fns)

duration = []
predicts = []
predicts_skip = []
voting_predicts = []
processed = 0
max_indoor_distance = 500.0
last_distance = 0.0
first_timestamp = float(sensors_data[0].split(",")[0])
times = []
classes = list()
cl_type = list()
times_nonv = []
times_visual = []
speed = []
speed_skip = []

pred_skip = 0
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
        cl_type.append(line.strip().split(' ')[2])

gt_data = []
predictions_to_return = 9
for i, im_fn in enumerate(im_fns):
    common.printProgressBar(i+1, im_fns_len, length=70, printEnd="\r")
    sensors_row = common.read_sensor_data(sensors_data, i)
    if sensors_row == None:
        continue
    distance = sensors_row[4]
    if n_gt < len(gt_ts) and os.path.basename(im_fn) == gt_ts[n_gt].split(",")[0]:
        c_gt =  int(gt_ts[n_gt].split(",")[1])
        n_gt += 1  
    gt_data.append(c_gt)
    decision, dt = common.distance_decision(distance, last_distance, max_indoor_distance)
    triggers.append(1 - decision)
    times_nonv.append(dt)
    #if decision:
        #voting_predicts.append(vote)
        #predicts.append(last_pr)
        #error.append(abs(last_pr - c_gt))
        #last_distance = distance
        #times.append((sensors_row[0] - first_timestamp)/(10**9))
        #continue
    #voting_predicts.append(vote)
    #predicts.append(pr)
    #error.append(abs(pr - c_gt))
    #last_distance = distance
    #times.append((sensors_row[0] - first_timestamp)/(10**9))
        #continue
    last_distance = distance
    image = Image.open(im_fn)
    image = np.array(image, dtype=np.uint8)
    image = cv2.resize(image[80:-80,:], (128, 128))
    #print(image.shape)
    image = Image.fromarray(np.uint8(image))
    image = loader(image).float()
    image = image.unsqueeze(0) 
    start = time.time()
    times.append((sensors_row[0] - first_timestamp)/(10**9))
    start_vis = time.time()
    #preds = net(image.cuda()).cpu()
    #preds = net(image.cuda())
    #print(preds)
    preds = net(image.cuda()).cpu().argmax()
    times_visual.append(time.time() - start_vis)
    #top_preds = np.argsort(preds)[::-1][0:predictions_to_return]
    # output the prediction
    #print(1 - preds.detach().numpy())
    predicts.append(preds.detach().numpy())
    speed.append(times_nonv[-1] + times_visual[-1])
    if decision:
        predicts_skip.append(pred_skip)
        speed_skip.append(times_nonv[-1]) 
    else:
        predicts_skip.append(predicts[-1])        
        pred_skip = predicts[-1]
        speed_skip.append(times_nonv[-1] + times_visual[-1])        
    last_pr = predicts[-1]    
    error.append(abs(predicts[-1] - c_gt))
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

f = open(os.path.join(path, "speed_distance_xception_gpu_torch.txt"), 'w')

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
plt.plot(times, speed)
plt.plot(times, speed_skip)
plt.xlabel("record time [s]")
plt.ylabel("duration [s]")
plt.savefig(os.path.join(path, "speed.pdf"))

plt.figure(4, figsize=(10, 4))
plt.plot(times, np.cumsum(speed))
plt.plot(times, np.cumsum(speed_skip))
plt.xlabel("record time [s]")
plt.ylabel("duration [s]")
plt.savefig(os.path.join(path, "speed_cumulative.pdf"))

plt.figure(2, figsize=(10, 4))
plt.plot(times, predicts)
plt.plot(times, gt_data)
plt.xlabel("time [s]")
plt.yticks([0,1], ["indoor", "outdoor"])
plt.savefig(os.path.join(path, "predicts.pdf"))

plt.figure(3, figsize=(10, 4))
plt.plot(times, predicts_skip)
plt.plot(times, gt_data)
plt.xlabel("time [s]")
plt.yticks([0,1], ["indoor", "outdoor"])
plt.savefig(os.path.join(path, "predicts_skip.pdf"))

plt.figure(5, figsize=(10, 4))
plt.plot(times, gt_data)
plt.xlabel("time [s]")
plt.yticks([0,1], ["indoor", "outdoor"])
plt.savefig(os.path.join(path, "groundtruth.pdf"))

plt.figure(6, figsize=(10, 4))
plt.plot(times, triggers)
plt.xlabel("time [s]")
#plt.yticks([0,1], ["indoor", "outdoor"])
plt.savefig(os.path.join(path, "triggers_distance.pdf"))

print(sum(error)/im_fns_len, sum(error), im_fns_len, np.mean(times_nonv), sum(times_nonv), np.mean(times_visual), sum(times_visual), np.mean(times_visual+times_nonv), sum(times_visual+times_nonv))

