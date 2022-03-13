import matplotlib.pyplot as plt
import numpy as np
import math
size=5
dosya = open("belgeler/sozluk.txt","r",encoding="utf-8")
lines = dosya.readlines()
count = len(lines)
if count%size==0:
    xpoints = np.array(list(range(size,count,size)))
    ypoints = np.zeros(len(list(range(size,count,size))))
    divides = np.zeros(len(list(range(size,count,size))))
else:
    xpoints = np.array(list(range(size,count+size,size)))
    ypoints = np.zeros(len(list(range(size,count,size)))+1)
    divides = np.zeros(len(list(range(size,count,size)))+1)
sayac = 0
for i in lines:
    table_index = math.floor(sayac/size)
    ypoints[table_index]+=int(i.split("-")[0])
    sayac+=1
divides.fill(size)
if count%size!=0:
    divides[-1]=count%size
divided =np.divide(ypoints,divides).astype(int)
plt.plot(xpoints, divided)
plt.show()
