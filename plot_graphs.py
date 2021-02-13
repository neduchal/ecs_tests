import os
import matplotlib.pyplot as plt
import numpy as np
import copy


path = "/media/neduchal/data/data/disertace/record2255"

data = open(os.path.join(path, "output.txt"), 'r').read().split('\n')

temperature = []
humidity = []
pressure = []
distance = []
timestamps = []

first_timestamp = float(data[0].split(",")[0])


for i, row in enumerate(data):
    if len(row) == 0:
        continue
    row_arr = row.split(",")
    if row_arr[1] == "[]":
        continue
    timestamps.append((float(row_arr[0]) - first_timestamp)/(10**9))
    temperature.append(float(row_arr[1]))
    humidity.append(float(row_arr[2]))
    pressure.append(float(row_arr[3])/1000.0)
    if len(row_arr) == 5:
        distance.append(float(row_arr[4]))

plt.figure(1, figsize=(10, 4))
plt.plot(timestamps, temperature)
plt.xlabel("time [s]")
plt.ylabel("temperature [°C]")
plt.savefig(os.path.join(path, "temperature.pdf"))

plt.figure(2, figsize=(10, 4))
plt.plot(timestamps, humidity)
plt.xlabel("time [s]")
plt.ylabel("humidity [%]")
plt.savefig(os.path.join(path, "humidity.pdf"))

plt.figure(3, figsize=(10, 4))
plt.plot(timestamps, pressure)
plt.xlabel("time [s]")
plt.ylabel("pressure [kPa]")
plt.savefig(os.path.join(path, "pressure.pdf"))

if data[0].split(",")[1] == "[]":
    plt.figure(4, figsize=(10, 4))
    plt.plot(timestamps, distance)
    plt.xlabel("time [s]")
    plt.ylabel("distance [cm]")
    plt.savefig(os.path.join(path, "distance.pdf"))
plt.show()
