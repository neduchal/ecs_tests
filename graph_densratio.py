from densratio import densratio
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


res = []

#Changepoint detection with the Pelt search method
v = temp[0:100]
indexes = list(range(0,100))
densratio_obj = densratio(indexes, v, 0.1)



for i, item in enumerate(temp):
    if i < 40: 
        continue
    start = time.time()
    #algo.fit(temp[i-40:i])
    #res.append(algo.predict(pen=4))
    res.append(densratio_obj.compute_density_ratio(temp[i]))
    times.append(time.time() - start)

resnp = np.array(res)


resdt = np.abs(resnp[1:] - resnp[0:-1])


print("Average time: %f, SUM: %f" % (np.mean(times), np.sum(times)))

plt.plot(temp)

plt.figure()

plt.plot(res)

plt.figure()

plt.plot(resdt)




plt.show()
