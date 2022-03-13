import numpy as np
import datetime
def calculate_std():
    dt = datetime.datetime.now()
    dosya=open("belgeler/sozluk.txt","r")
    lines=dosya.readlines()
    dosya.close()
    sayac=0
    total=0
    arr = np.zeros(len(lines))
    for i in lines:
        arr[sayac]=int(lines[sayac].split("-")[0])
        sayac+=1
    std = np.std(arr,0)
    mean = np.mean(arr)
    dosya2=open("belgeler/std_history.txt","a")
    dosya2.write(str(dt)+"  -  std: "+str(std)+"  -  mean:"+str(mean)+"\n")
    dosya2.close()
    return std,mean
if __name__=="__main__":
    print(calculate_std())
