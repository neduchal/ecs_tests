import bocd
import numpy as np
import matplotlib.pyplot as plt
import time

data = open("output1_1.txt", 'r').read().split('\n')


times= []

temp = []
distance = []

for line in data:
    linea = line.split(",")
    if len(linea) < 4:
	    continue
    if np.isnan(np.double(linea[1])):
        temp.append(temp[-1])
        distance.append(distance[-1])
    else:
        temp.append(np.double(linea[1]))
        distance.append(np.double(linea[4]))

temp = np.array(temp[0:12000])

bc = bocd.BayesianOnlineChangePointDetection(bocd.ConstantHazard(500), bocd.StudentT(mu=0, kappa=1, alpha=1, beta=1))

res = []

for i, item in enumerate(temp):
    start = time.time()
    bc.update(item)
    times.append(time.time() - start)
    res.append(bc.rt)

resnp = np.array(res)


resdt = np.abs(resnp[1:] - resnp[0:-1])


print("Average time: %f, SUM: %f" % (np.mean(times), np.sum(times)))

plt.plot(temp)

plt.figure()

plt.plot(res)

plt.figure()

plt.plot(resdt)

plt.show()
