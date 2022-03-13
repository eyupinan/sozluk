import soundfile as sf
import sounddevice as sd
import os
import numpy as np
from scipy.io.wavfile import write
import math 

index=0
index1=0
index2=0
liste=[]
liste2=[]
liste1=[]
sayac=0
while True:
	if os.path.isdir("./eng_voices/"+str(index1)+"_"+str(index2)):
		for t in os.listdir("./eng_voices/"+str(index1)+"_"+str(index2)):
			if index2==0:
				ng, fs=  sf.read("./eng_voices/"+str(index1)+"_"+str(index2)+"/"+t, dtype='float32')
				print("index:"+str(index)+" file:"+str(index1)+"_"+str(index2)+"/"+t+" ort var:" ,np.mean(ng),np.var(ng))
			else:
				ng, fs=  sf.read("./eng_voices/"+str(index1)+"_"+str(index2)+"/"+t, dtype='float32')
				print("index:"+str(index)+" file:"+str(index1)+"_"+str(index2)+"/"+t+" ort var:" ,np.mean(ng),np.var(ng))

			if np.var(ng)<0.002:
				print("new :"+str(index)+" file:"+str(index1)+"_"+str(index2)+"/"+t+" ort var:" ,np.mean(ng*3),np.var(ng*3))
				#write("./eng_voices/"+str(index1)+"_"+str(index2)+"/"+t, 44100, ng*3)
			liste.append(ng)
			if sayac<300:
				liste1.append(ng)
			else:
				#write("./eng_voices/"+str(index1)+"_"+str(index2)+"/"+t, 44100,ng/0.55710626)
				liste2.append(ng)
			index+=1

	else:
		if index2==0:
			break
		index1+=1
		index2=0
		continue
	index2+=1
	sayac+=1
total_arr= np.array(liste)
total_arr_1= np.array(liste1)
total_arr_2= np.array(liste2)
"""liste=[]
sayac=0
for t in os.listdir("./test"):
	print(t)
	ng, fs=  sf.read("./test/"+t, dtype='float32')
	print(np.mean(ng),np.var(ng))
	print("new:",np.mean(ng),np.var(ng))
	liste.append(ng)
	sayac+=1
	#if sayac>=47:
		
	#write("./test/"+t, 44100, ng*1.2)
total_arr2= np.array(liste)"""
print("son dataset:",np.mean(total_arr),np.var(total_arr),np.std(total_arr))
#print("son dataset:",list(np.mean(total_arr,axis=(1,2))),list(np.var(total_arr,axis=(1,2))))
print("son dataset:",np.mean(total_arr),np.var(total_arr_1),np.std(total_arr_1))
print("son dataset:",np.mean(total_arr),np.var(total_arr_2),np.std(total_arr_2))
print("güncellenmiş:",np.var((total_arr_2/(np.std(total_arr_2)/np.std(total_arr_1)))))
print("bölen:",(np.std(total_arr_2)/np.std(total_arr_1)))
#print("son test:",list(np.mean(total_arr2,axis=(1,2))),list(np.var(total_arr2,axis=(1,2))))
