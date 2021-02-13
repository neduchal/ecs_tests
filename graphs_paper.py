import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy import signal
from scipy.interpolate import splrep, splev
from cv_bridge import CvBridge, CvBridgeError
import cv2

def cusum(x,mean=0,K=0):
    """Tabular CUSUM per Montgomery,D. 1996 "Introduction to Statistical Process Control" p318 
    x    : series to analyze
    mean : expected process mean
    K    : reference value, allowance, slack value-- suggest K=1/2 of the shift to be detected.

    Returns:
    x  Original series
    Cp positive CUSUM
    Cm negative CUSUM
    """
    Cp=(x*0).copy()
    Cm=Cp.copy()
    for ii in np.arange(len(x)):
        if ii == 0:
            Cp[ii]=x[ii]
            Cm[ii]=x[ii]
        else:
            Cp[ii]=np.max([0,x[ii]-(mean+K)+Cp[ii-1]])
            Cm[ii]=np.max([0,(mean-K)-x[ii]+Cm[ii-1]])
    return({'x':x, 'Cp': Cp, 'Cm': Cm})

def derivation_ratio(xpar, dx1):
    x = xpar.copy()
    #dx1 = np.mean(x[-39:] - x[-40:-1])
    dx2 = np.mean(x[-9:] - x[-10:-1])
    if (dx2 == 0): 
        return 1   
    ratio = np.abs((dx1+0.03)/np.float(dx2+0.03))
    if (ratio == 0): 
        return 1    
    return np.max([ratio, 1/ratio])


data = open("output1.txt", 'r').read().split('\n')
#data = open("sensors.txt", 'r').read().split('\n')
temp = []
humidity = []
distance = []

for line in data:
    linea = line.split(",")
    if len(linea) < 4:
	    continue
    if np.isnan(np.double(linea[1])):
        temp.append(temp[-1])
    else:
        temp.append(np.double(linea[1]))

temp = np.array(temp)

t = np.arange(len(temp))/(60.0 * 24.0)

plt.figure(1, [10,3])
plt.plot(t,temp)
plt.xlabel("time [min]")
plt.ylabel("temperature")


temp_df = temp[1:] - temp[0:-1]

temp_df_1s = []

for k in range(0, len(temp_df), 24):
    temp_df_1s.append(np.sum(temp_df[k:k+24]))

plt.figure(2, [10,3])
plt.plot(np.arange(len(temp_df_1s))/60.0, temp_df_1s)
plt.xlabel("time [min]")
plt.ylabel("temperatures difference 1s")

plt.figure(4, [10,3])
plt.plot(t[1:],temp_df)
plt.xlabel("time [min]")
plt.ylabel("temperatures difference")

cusumres = []
for i in range(30, len(temp)):
    avr = np.mean(temp[i-30:i])
    cs = cusum(temp[i-30:i], mean=avr, K=1/2)
    cusumres.append(np.max(cs["Cp"] - cs["Cm"]))

dx_ratio = [1]
transitions_dx = []
n = 0
for i in range(40, len(temp)):
    if i == 40:
        dx1 = np.mean(temp[-39:] - temp[-40:-1])
        n = 39
    else:
        #print (dx1 * n, temp[i] - temp[i-1], float(n+1), ((dx1 * n) + (temp[i] - temp[i-1])) / float(n+1) )
        dx1 = ((dx1 * n) + (temp[i] - temp[i-1])) / float(n+1)    
        if np.isnan(dx1):
            break
        n = n+1
    #print(dx1)
    dx_ratio.append(derivation_ratio(temp[i-10:i], dx1))
    if dx_ratio[-1] > 1.5:    
        dx1 = np.mean(temp[i-39:i] - temp[i-40:i-1])
        n = 39
        transitions_dx.append(1)
    else:
        transitions_dx.append(0)
        #print(dx1)


plt.figure(5, [10,3])
plt.plot(t[:-30], cusumres)
plt.xlabel("time [min]")
plt.ylabel("CUSUM value")

plt.figure(8, [10,3])
plt.plot(t[:-30], (np.array(cusumres) > 1))
plt.xlabel("time [min]")
plt.ylabel("CUSUM triggers")


plt.figure(6, [10,3])
plt.plot(t[39:], dx_ratio)
plt.xlabel("time [min]")
plt.ylabel("DiRat value")

plt.figure(7, [10,3])
plt.plot(t[40:], transitions_dx)
plt.xlabel("time [min]")
plt.ylabel("trigger value")

#e_temp = 21.669
#var_temp = np.double(0.110)

temp2= temp[3600:4300]

e_temp = np.mean(temp2)
var_temp = np.std(temp2)

e_temp2 = []
var_temp2 = []
for k in range(0, len(temp),1):
    e_temp2.append(np.mean(temp[k:k+10]))
    var_temp2.append(np.std(temp[k:k+10]))

#plt.figure(11, [10,3])
#plt.plot(e_temp2)
#plt.xlabel("time [min]")
#plt.ylabel("temperature")

plt.figure(12, [10,3])
plt.plot(t, var_temp2)

plt.xlabel("time [min]")
plt.ylabel("variance of signal within window")

e = (var_temp2 > (var_temp))
for i in range(1, len(e)):
     if sum(e[i-24:i]) > 0:
        e[i] = 0

plt.figure(13, [10,3])
plt.plot(t, e)
plt.xlabel("time [min]")
plt.ylabel("trigger value")
#e1 = (var_temp2 > (2.5 * var_temp)) * 0.05
#e2 = (var_temp2 > (3.0 * var_temp)) * 0.05
#e3 = (var_temp2 > (3.5 * var_temp)) * 0.05

#e = e[1:] - e[:-1]

#plt.plot(e)


edf = [0]
for i in range(1, len(temp_df_1s)):
    edf.append(temp_df_1s[i] - temp_df_1s[i-1])
print(type(e),type(edf))
edf = np.array(edf)
#plt.plot(edf[1:]-edf[0:-1])
#plt.plot(e1)
#plt.plot(e2)
#plt.plot(e3)


 

'''
temp_df_multi = copy.deepcopy(temp_df)

temp_df_1s = []

for t in range(0, len(temp_df), 10):
    temp_df_1s.append(np.sum(temp_df[t:t+10]))
    temp_df_multi[t] += np.sum(temp_df[t:t+10])


plt.figure(5, [10,3])
plt.plot(temp_df_1s)

temp_df_2s = []

for t in range(0, len(temp_df), 20):
    temp_df_2s.append(np.sum(temp_df[t:t+20]))
    temp_df_multi[t] += np.sum(temp_df[t:t+20])

plt.figure(6, [10,3])
plt.plot(temp_df_2s)

temp_df_5s = []

for t in range(0, len(temp_df), 50):
    temp_df_5s.append(np.sum(temp_df[t:t+50]))
    temp_df_multi[t] += np.sum(temp_df[t:t+50])

plt.figure(7, [10,3])
plt.plot(temp_df_5s)

temp_df_10s = []

for t in range(0, len(temp_df), 100):
    temp_df_10s.append(np.sum(temp_df[t:t+100]))
    temp_df_multi[t] += np.sum(temp_df[t:t+100])

plt.figure(8, [10,3])
plt.plot(temp_df_10s)


plt.figure(9, [10,3])
plt.plot(temp_df_multi)


'''
#plt.figure(2, [10,3])
#plt.plot(humidity)

#plt.figure(3, [10,3])
#plt.plot(distance)


'''
humidity = np.array(humidity)
temp = np.array(temp)
from scipy.interpolate import splrep, splev
f = splrep(range(len(temp)),temp,k=5,s=5)
temp = splev(range(len(temp)), f)

f2 = splrep(range(len(humidity)),humidity,k=5,s=5)
humidity = splev(range(len(humidity)), f2)
hum_df = humidity[1:] - humidity[0:-1]
hum_transformed = []

pres_df = pressure[1:] - pressure[0:-1]
pres_transformed = []

plt.figure(1, [10,3])
temp2 = temp[::100]
#plt.plot(temp2 - np.mean(temp2))
temp2_df = temp2[1:] - temp2[0:-1]
#plt.plot(range(1,len(temp2_df)+1),temp2_df, 'r*')
temp2_df2 = temp2_df[1:] - temp2_df[0:-1]
#plt.plot(range(2,len(temp2_df)+1),temp2_df2, 'g.')
res = np.abs(temp2_df2[0:]*temp2_df[1:])
plt.plot(range(1,len(temp2_df)), res , 'm')
plt.plot(range(1,len(temp2_df)), res > max(res)*0.3, 'r')
plt.ylabel("Hodnota detekce")
plt.xlabel("Cas [100 s]")

plt.figure(2, [10,3])
plt.plot(temp)
plt.ylabel("Teplota")
plt.xlabel("Cas [desetiny s]")
'''



'''
plt.figure(2)
hum2 = humidity[::100]
#plt.plot(hum2 - np.mean(hum2))
hum2_df = hum2[1:] - hum2[0:-1]
#plt.plot(range(1,len(hum2_df)+1),hum2_df, 'r*')
hum2_df2 = hum2_df[1:] - hum2_df[0:-1]
#plt.plot(range(2,len(hum2_df)+1),hum2_df2, 'g.')
res = np.abs(hum2_df2[0:]*hum2_df[1:])
plt.plot(range(1,len(hum2_df)), res, 'm')
plt.plot(range(1,len(hum2_df)), res > (max(res)*0.05), 'r')
'''

'''
plt.figure(3)
plt.plot(hum_transformed)

plt.figure(4)
plt.plot(humidity)

plt.figure(5)
plt.plot(pres_transformed)

plt.figure(6)
plt.plot(pressure)
'''
plt.show()




