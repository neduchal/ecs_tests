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
#import centrist
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

net = models.vgg16_bn(pretrained=False)
num_ftrs = net.classifier[6].in_features
net.classifier[6] = nn.Linear(num_ftrs,2)

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
#net.avg_pool = nn.AvgPool2d(kernel_size=2, stride =2)
#num_ftrs = net.last_linear.in_features
#net.last_linear = nn.Linear(num_ftrs, num_classes)


#net = models.wide_resnet50_2(pretrained=True)
#num_ftrs = net.fc.in_features
#net.fc = nn.Linear(num_ftrs, num_classes)

#net = models.resnext50_32x4d(pretrained=True)
#num_ftrs = net.fc.in_features
#net.fc = nn.Linear(num_ftrs, num_classes)

#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#net = pretrainedmodels.xception()
#num_ftrs = net.last_linear.in_features
#net.last_linear = nn.Linear(num_ftrs, num_classes)
#set_parameter_requires_grad(net, feature_extract)
#normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomResizedCrop(size=size, scale=(0.5,1.0), ratio=(1.0,1.0)), transforms.ToTensor(), normalize])

net.load_state_dict(torch.load('models/VGG16_best.pth'))
#net.load_state_dict(torch.load('models/DenseNet_best.pth'))
#net.load_state_dict(torch.load('models/InceptionResNetv2_best.pth'))
#net.load_state_dict(torch.load('models/InceptionV4_best.pth'))
#net.load_state_dict(torch.load('models/Xception_best.pth'))
#net.load_state_dict(torch.load('models/ResNetWide_best.pth'))
#net.load_state_dict(torch.load('models/ResNext_best.pth'))

net.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

path = "/home/neduchal/Dokumenty/Data/disertace/record0949"
sensors_data = open(os.path.join(path, "output.txt"), 'r').read().split('\n')
file_name = 'categories_places365_io.txt'

#gt_ts = open(os.path.join(path, "gt.txt"), "r").read().split("\n")
c_gt = 0
n_gt = 0
error = []

loader = transforms.Compose([ transforms.ToTensor()])

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
    image = cv2.resize(image, (128,128))
    #print(image.shape)
    image = Image.fromarray(np.uint8(image))
    image = loader(image).float()
    image = image.unsqueeze(0) 
    start = time.time()
    times.append((sensors_row[0] - first_timestamp)/(10**9))
    start_vis = time.time()
    #preds = net(image.cuda()).cpu()
    preds = net(image.cuda()).cpu().argmax() 
    #preds = net(image.cuda())
    times_visual.append(time.time() - start_vis)
    # output the prediction 
    predicts.append(preds.detach().numpy())
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

print(sum(error)/im_fns_len, sum(error), im_fns_len, np.mean(times_nonv), sum(times_nonv), np.mean(times_visual), sum(times_visual), np.mean(times_visual+times_nonv), sum(times_visual+times_nonv))
