import tensorflow as tf
import numpy as np
import sys
import vlc
import sounddevice as sd
import soundfile as sf
import librosa
import threading
from threading import Thread
from scipy.io.wavfile import write
import os
from multiprocessing import Process,Array
import multiprocessing
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import random
from show import show_diff,show_same
import re
from tkinter import *
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.python.keras.backend import set_session
from tensorflow.keras import activations
from tensorflow.keras import losses
tf.compat.v1.disable_eager_execution()
tf.debugging.set_log_device_placement(True)
tf.keras.backend.set_image_data_format('channels_last')
def sigmoid(params):
        return 1/(1+np.exp(-1*params))
def my_loss(weight):
        def weighted_cross_entropy_with_logits(labels, logits):
                loss = tf.nn.weighted_cross_entropy_with_logits(
                        labels, logits, weight
                )
                return loss
        return weighted_cross_entropy_with_logits
def initializer(kernel_sizes,init="random_normal",activation="relu"):
        if init=="random_normal":
            std=0.03
        if init=="xavier_normal":
            fan_in=np.prod(kernel_sizes[:-1])
            fan_out=kernel_sizes[-1]
            if activation=="softmax" or activation=="sigmoid" :
                std = np.sqrt(2. / (fan_in + fan_out))
            elif activation=="relu":
                std = np.sqrt(2. / (fan_in))
        if init=="xavier_uniform":
            fan_in=np.prod(kernel_sizes[:-1])
            fan_out=kernel_sizes[-1]
            if activation=="softmax" or activation=="sigmoid":
                std = np.sqrt(6. / (fan_in + fan_out))
            elif activation=="relu":
                std = np.sqrt(6. / (fan_in))
        return std
def manipulate(data, noise_factor):
    noise = np.random.randn(len(data),len(data[0]))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data
def manipulate_shift(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif self.shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data
def complex_aug(data):
        trans=np.transpose(data)
        SAMPLE_RATE=44100
        
        augment = Compose([
        AddGaussianNoise(min_amplitude=random.uniform(0.008, 0.016), max_amplitude=random.uniform(0.022, 0.050), p=0.65),
        TimeStretch(min_rate=0.7, max_rate=1.15, p=0.55),
        PitchShift(min_semitones=-5, max_semitones=5, p=0.53),
        Shift(min_fraction=-0.6, max_fraction=0.6, p=0.45),
        ])
        augmented_samples = augment(samples=trans, sample_rate=SAMPLE_RATE)
        return np.transpose(augmented_samples)
def complex_aug2(data):
        trans=np.transpose(data)
        SAMPLE_RATE=44100
        augment = Compose([
        AddGaussianNoise(min_amplitude=random.uniform(0.00065, 0.00095), max_amplitude=random.uniform(0.016, 0.025), p=0.45),
        TimeStretch(min_rate=random.uniform(0.91, 0.94), max_rate=random.uniform(1.45, 1.70), p=0.45),
        PitchShift(min_semitones=-5, max_semitones=5, p=0.45),
        Shift(min_fraction=-0.55, max_fraction=0.55, p=0.55),
        ])
        augmented_samples = augment(samples=trans, sample_rate=SAMPLE_RATE)
        return np.transpose(augmented_samples)
def complex_aug3(data):
        trans=np.transpose(data)
        SAMPLE_RATE=44100
        augment = Compose([
        AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.04, p=0.6),
        TimeStretch(min_rate=0.9, max_rate=1.40, p=0.55),
        PitchShift(min_semitones=random.uniform(-7, -5), max_semitones=random.uniform(5, 7), p=0.55),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.45),
        ])
        augmented_samples = augment(samples=trans, sample_rate=SAMPLE_RATE)
        return np.transpose(augmented_samples)
def complex_aug4(data):
        trans=np.transpose(data)
        SAMPLE_RATE=44100
        augment = Compose([
        AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.015, p=0.35),
        TimeStretch(min_rate=0.85, max_rate=1.25, p=0.60),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.55),
        Shift(min_fraction=random.uniform(-1, -0.4), max_fraction=random.uniform(0.4, 1), p=0.65),
        ])
        augmented_samples = augment(samples=trans, sample_rate=SAMPLE_RATE)
        return np.transpose(augmented_samples)
def manipulate_speed(data, speed_factor=0.1):
    return librosa.effects.time_stretch(data, speed_factor)
def loss_function(y_,y=None,loss_type="kategori"):#y_ prediction ,y ground truth
        y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
        if loss_type=="kategori":
                
                arg=tf.cast(tf.argmax(y,1),tf.int32)
                arg=tf.expand_dims(arg,axis=1)
                y_indexes=tf.concat([tf.expand_dims(tf.range(tf.shape(y)[0]),axis=1),arg],axis=1)
                gathered=tf.gather_nd(y_clipped,y_indexes)
                log_likelihood = -tf.math.log(gathered)/tf.cast(tf.shape(y)[0],tf.float32)
                cross_entropy = tf.reduce_sum(log_likelihood)
        if loss_type=="binary":
                cross_entropy = tf.reduce_mean(-(tf.reduce_sum(y * tf.math.log(y_clipped)
                                    + (1 - y) * tf.math.log(1 - y_clipped), axis=1)))
        ##
        return cross_entropy
def shuffling(data,label):
        seed = np.random.randint(0, 100000)  

        np.random.seed(seed)  
        np.random.shuffle(data)  
        np.random.seed(seed)  
        np.random.shuffle(label)
        return data,label
def f1(data_unshaped_main,return_dict,nm,shape,dtype):
        try:
                dosya2 = open("C:/Users/eyyup/Desktop/deger.txt","a+")
                #existing_shm = shared_memory.SharedMemory(name=name)
                #data_unshaped_main = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
                dosya2.write("1:"+str(np.var(data_unshaped_main))+"\n")
                dosya2.close()
                length= len( data_unshaped_main)
                data_unshaped=np.zeros((length,110250,2),np.float32)
                sayac=0
                for i in range(length):
                            new_data=complex_aug4(data_unshaped_main[i]*1.25)
                            rand = random.randint(0,int(new_data.shape[0]/2))
                            part1=np.copy(new_data[rand:])
                            part2=np.copy(new_data[:rand])
                            new_data[0:new_data.shape[0]-rand]=np.copy(part1)
                            new_data[new_data.shape[0]-rand:]=np.copy(part2)
                            carpan = random.randint(1000,4000)
                            new_data = new_data * (carpan/1000)
                            if i==0:
                                    write("./mp3/noise8.wav", 44100, np.array(new_data).astype(np.float32))

                            data_unshaped[sayac]=new_data
                            sayac+=1
                return_dict[nm]=data_unshaped
                return_dict[nm+"-std"]=np.std(data_unshaped)
                return_dict[nm+"-mean"]=np.mean(data_unshaped)
                #existing_shm.close()
        except Exception as err:
                dosya = open("C:/Users/eyyup/Desktop/hata.txt","a+")
                dosya.write(str(err)+"\n")
                dosya.close()
                
                
def f2(data_unshaped_main,return_dict ,nm,shape,dtype):
        try:
                dosya2 = open("C:/Users/eyyup/Desktop/deger.txt","a+")
                #existing_shm = shared_memory.SharedMemory(name=name)
                #data_unshaped_main = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
                dosya2.write("2:"+str(np.var(data_unshaped_main))+"\n")
                dosya2.close()
                length= len( data_unshaped_main)
                data_unshaped=np.zeros((length,110250,2),np.float32)
                sayac=0
                for i in range(length):
                            new_data=complex_aug(data_unshaped_main[i])
                            rand = random.randint(0,int(new_data.shape[0]/2))
                            part1=np.copy(new_data[rand:])
                            part2=np.copy(new_data[:rand])
                            new_data[0:new_data.shape[0]-rand]=np.copy(part1)
                            new_data[new_data.shape[0]-rand:]=np.copy(part2)
                            carpan = random.randint(1000,2000)
                            new_data = new_data * (carpan/1000)
                            if i==0:
                                    
                                    write("./mp3/noise5.wav", 44100, new_data.astype(np.float32))
                            data_unshaped[sayac]=new_data
                            sayac+=1
                return_dict[nm]=data_unshaped
                return_dict[nm+"-std"]=np.std(data_unshaped)
                return_dict[nm+"-mean"]=np.mean(data_unshaped)
                #existing_shm.close()
        except Exception as err:
                dosya = open("C:/Users/eyyup/Desktop/hata.txt","a+")
                dosya.write("2: "+str(err)+"\n")
                dosya.close()
def f3(data_unshaped_main,return_dict ,nm,shape,dtype):
        try:
                dosya2 = open("C:/Users/eyyup/Desktop/deger.txt","a+")
                #existing_shm = shared_memory.SharedMemory(name=name)
                #data_unshaped_main = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
                dosya2.write("3:"+str(np.var(data_unshaped_main))+"\n")
                dosya2.close()
                length= len( data_unshaped_main)
                data_unshaped=np.zeros((length,110250,2),np.float32)
                sayac=0
                for i in range(length):
                            new_data=complex_aug2(data_unshaped_main[i])
                            rand = random.randint(0,int(new_data.shape[0]/2))
                            part1=np.copy(new_data[rand:])
                            part2=np.copy(new_data[:rand])
                            new_data[0:new_data.shape[0]-rand]=np.copy(part1)
                            new_data[new_data.shape[0]-rand:]=np.copy(part2)
                            carpan = random.randint(750,2200)
                            new_data = new_data * (carpan/1000)
                            if i==0:
                                    
                                    write("./mp3/noise6.wav", 44100, new_data.astype(np.float32))
                            data_unshaped[sayac]=new_data
                            sayac+=1
                return_dict[nm]=data_unshaped
                return_dict[nm+"-std"]=np.std(data_unshaped)
                return_dict[nm+"-mean"]=np.mean(data_unshaped)
                #existing_shm.close()
        except Exception as err:
                dosya = open("C:/Users/eyyup/Desktop/hata.txt","a+")
                dosya.write("3: "+str(err)+"\n")
                dosya.close()
def f4(data_unshaped_main,return_dict,nm,shape,dtype ):
        try:
                dosya2 = open("C:/Users/eyyup/Desktop/deger.txt","a+")
                #existing_shm = shared_memory.SharedMemory(name=name)
                #data_unshaped_main = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
                dosya2.write("4:"+str(np.var(data_unshaped_main))+"\n")
                dosya2.close()
                length= len( data_unshaped_main)
                data_unshaped=np.zeros((length,110250,2),np.float32)
                sayac=0
                for i in range(length):
                            new_data=complex_aug3(data_unshaped_main[i])
                            rand = random.randint(0,int(new_data.shape[0]/2))
                            part1=np.copy(new_data[rand:])
                            part2=np.copy(new_data[:rand])
                            new_data[0:new_data.shape[0]-rand]=np.copy(part1)
                            new_data[new_data.shape[0]-rand:]=np.copy(part2)
                            carpan = random.randint(1000,3000)
                            new_data = new_data * (carpan/1000)
                            if i==0:
                                    
                                    write("./mp3/noise7.wav", 44100, new_data.astype(np.float32))
                            data_unshaped[sayac]=new_data
                            sayac+=1
                return_dict[nm]=data_unshaped
                return_dict[nm+"-std"]=np.std(data_unshaped)
                return_dict[nm+"-mean"]=np.mean(data_unshaped)
                #existing_shm.close()
        except Exception as err:
                dosya = open("C:/Users/eyyup/Desktop/hata.txt","a+")
                dosya.write("4: "+str(err)+"\n")
                dosya.close()
class training:
        def __init__(self,control_window_bool=True):
                self.source="tr_voices"
                self.control_window_bool=control_window_bool
                self.dictionary={}
                self.kernels=[]
                self.batch_mean_list=[]
                self.batch_std_list=[]
                self.batch_mean_plc_list=[]
                self.batch_std_plc_list=[]
                self.y_list=[]
                self.b_list=[]
                self.stop=False
                self.isTraining= tf.compat.v1.placeholder(tf.bool, shape=())
                
        def control_window(self):
                self.tk=Tk()
                self.entry = Entry()
                self.button = Button(text="evaluate",command=lambda:self.evaluate())
                self.entry.pack()
                self.button.pack()
        def evaluate(self):
                code = self.entry.get()
                if re.search("import", code) or re.search("sys.", code) or re.search("os\.", code) or re.search("Process", code) or re.search("Array", code) or re.search("multiprocessing.", code):
                        self.entry.delete(0, 'end')
                else:
                        exec(code)
                        self.entry.delete(0, 'end')
        def fill_one_hot(self,index):
                empty_copy=np.copy(self.empty)
                empty_copy[index]=1
                return empty_copy
        def get_data(self,training=True):
                self.empty=[]
                self.output_count=0
                for q in os.listdir("./"+self.source+"/"):
                        self.empty.append(0)
                        self.output_count+=1
                self.empty=np.array(self.empty)
                index=0
                index2=0
                if training==True:
                        self.one_hot_main=[]
                        self.one_hot_test=[]
                sayac=0
                index=0
                index1=0
                index2=0
                source=self.source
                f=False
                t=0
                while True:
                        if os.path.isdir("./"+source+"/"+str(index1)+"_"+str(index2)):
                                data_count=0
                                for t in os.listdir("./"+source+"/"+str(index1)+"_"+str(index2)):
                                        
                                        if training==True:
                                                if data_count<1:
                                                        
                                                        if sayac==0:
                                                                ng, fs=  sf.read("./"+source+"/"+str(index1)+"_"+str(index2)+"/"+t, dtype='float32')
                                                                print("index:"+str(sayac)+" ort var:" ,np.mean(ng),np.var(ng))
                                                                data=np.reshape(ng,(1,ng.shape[0],ng.shape[1]))
                                                        else:
                                                                ng, fs=  sf.read("./"+source+"/"+str(index1)+"_"+str(index2)+"/"+t, dtype='float32')
                                                                print("index:"+str(sayac)+" ort var:" ,np.mean(ng),np.var(ng))
                                                                data=np.append(data,np.reshape(ng,(1,ng.shape[0],ng.shape[1])),axis=0)
                                                        self.one_hot_main.append(self.fill_one_hot(index))
                                                
                                                else:
                                                        if sayac==0:
                                                                ng, fs=  sf.read("./"+source+"/"+str(index1)+"_"+str(index2)+"/"+t, dtype='float32')
                                                                print("index:"+str(sayac)+" ort var:" ,np.mean(ng),np.var(ng))
                                                                self.test_data=np.reshape(ng,(1,ng.shape[0],ng.shape[1]))
                                                        else:
                                                                ng, fs=  sf.read("./"+source+"/"+str(index1)+"_"+str(index2)+"/"+t, dtype='float32')
                                                                print("index:"+str(sayac)+" ort var:" ,np.mean(ng),np.var(ng))
                                                                try:
                                                                        self.test_data=np.append(self.test_data,np.reshape(ng,(1,ng.shape[0],ng.shape[1])),axis=0)
                                                                except:
                                                                        self.test_data=np.reshape(ng,(1,ng.shape[0],ng.shape[1]))
                                                        self.one_hot_test.append(self.fill_one_hot(index))
                                                

                                                        
                                        self.dictionary[index]=str(index1)+"_"+str(index2)
                                        #self.one_hot_main.append(self.fill_one_hot(index))
                                        sayac+=1
                                        data_count+=1
                                        
                                if sayac>100:
                                        None
                                index2+=1
                                index+=1
                                
                        else:
                                #print("else:",index1,index2)
                                if index2==0:
                                        if index1>=2000:
                                                break
                                index1+=1
                                index2=0
                        
                        
                        
                """for q in os.listdir("./voices/"):
                        self.dictionary[index]=q
                        for t in os.listdir("./voices/"+q):
                                if index2==0:
                                        ng, fs=  sf.read("./voices/"+q+"/"+t, dtype='float32')
                                        print("index:"+str(sayac)+" ort var:" ,np.mean(ng),np.var(ng))
                                        data=np.reshape(ng,(1,ng.shape[0],ng.shape[1]))
                                else:
                                        ng, fs=  sf.read("./voices/"+q+"/"+t, dtype='float32')
                                        print("index:"+str(sayac)+" ort var:" ,np.mean(ng),np.var(ng))

                                        data=np.append(data,np.reshape(ng,(1,ng.shape[0],ng.shape[1])),axis=0)

                                self.one_hot_main.append(self.fill_one_hot(index))
                                index2+=1
                                sayac+=1
                        index+=1"""
                if training!=False:
                        self.main_data_safe = data
                        return data         
        def augmentation(self,data_unshaped_main=None):
                try:
                        if data_unshaped_main==None:
                              data_unshaped_main=np.copy(self.main_data_safe)
                except:
                        None
                tekrar=9
                print(data_unshaped_main)
                y=tf.compat.v1.placeholder(tf.float32,[None,self.output_count])
                sz=len(data_unshaped_main)
                print(sz)
                sayac=sz
                data_unshaped=np.zeros((sz*tekrar,110250,2),np.float32)
                data_unshaped[:sz]=np.copy(data_unshaped_main)
                manager = multiprocessing.Manager()
                return_dict = manager.dict()
                print("main var :" ,np.var(data_unshaped_main))
                #shm = shared_memory.SharedMemory(create=True, size=data_unshaped_main.nbytes)
                #buffer = np.ndarray(data_unshaped_main.shape, dtype=data_unshaped_main.dtype, buffer=shm.buf)
                #buffer[:] = data_unshaped_main[:]
                
                p1_0 = Process(target=f1, args=(np.copy(data_unshaped_main[0:int(sz/2)]),return_dict,"1_0",data_unshaped_main.shape,data_unshaped_main.dtype))
                p1_1 = Process(target=f1, args=(np.copy(data_unshaped_main[int(sz/2):sz]),return_dict,"1_1",data_unshaped_main.shape,data_unshaped_main.dtype))
                sayac+=sz
                p2_0 = Process(target=f2, args=(np.copy(data_unshaped_main[0:int(sz/2)]),return_dict,"2_0",data_unshaped_main.shape,data_unshaped_main.dtype))
                p2_1 = Process(target=f2, args=(np.copy(data_unshaped_main[int(sz/2):sz]),return_dict,"2_1",data_unshaped_main.shape,data_unshaped_main.dtype))
                sayac+=sz
                p3_0 = Process(target=f3, args=(np.copy(data_unshaped_main[0:int(sz/2)]),return_dict,"3_0",data_unshaped_main.shape,data_unshaped_main.dtype))
                p3_1 = Process(target=f3, args=(np.copy(data_unshaped_main[int(sz/2):sz]),return_dict,"3_1",data_unshaped_main.shape,data_unshaped_main.dtype))
                sayac+=sz
                p4_0 = Process(target=f4, args=(np.copy(data_unshaped_main[0:int(sz/2)]),return_dict,"4_0",data_unshaped_main.shape,data_unshaped_main.dtype))
                p4_1 = Process(target=f4, args=(np.copy(data_unshaped_main[int(sz/2):sz]),return_dict,"4_1",data_unshaped_main.shape,data_unshaped_main.dtype))
                sayac+=sz
                p5_0 = Process(target=f3, args=(np.copy(data_unshaped_main[0:int(sz/2)]),return_dict,"5_0",data_unshaped_main.shape,data_unshaped_main.dtype))
                p5_1 = Process(target=f3, args=(np.copy(data_unshaped_main[int(sz/2):sz]),return_dict,"5_1",data_unshaped_main.shape,data_unshaped_main.dtype))
                sayac+=sz
                p6_0 = Process(target=f4, args=(np.copy(data_unshaped_main[0:int(sz/2)]),return_dict,"6_0",data_unshaped_main.shape,data_unshaped_main.dtype))
                p6_1 = Process(target=f4, args=(np.copy(data_unshaped_main[int(sz/2):sz]),return_dict,"6_1",data_unshaped_main.shape,data_unshaped_main.dtype))
                sayac+=sz
                p7_0 = Process(target=f1, args=(np.copy(data_unshaped_main[0:int(sz/2)]),return_dict,"7_0",data_unshaped_main.shape,data_unshaped_main.dtype))
                p7_1 = Process(target=f1, args=(np.copy(data_unshaped_main[int(sz/2):sz]),return_dict,"7_1",data_unshaped_main.shape,data_unshaped_main.dtype))
                sayac+=sz
                p8_0 = Process(target=f2, args=(np.copy(data_unshaped_main[0:int(sz/2)]),return_dict,"8_0",data_unshaped_main.shape,data_unshaped_main.dtype))
                p8_1 = Process(target=f2, args=(np.copy(data_unshaped_main[int(sz/2):sz]),return_dict,"8_1",data_unshaped_main.shape,data_unshaped_main.dtype))
                
                p1_0.start()
                print("p1_0 started")
                p1_1.start()
                print("p1_1 started")
                p2_0.start()
                print("p2_0 started")
                p2_1.start()
                print("p2_1 started")
                
                
                p1_0.join()
                print("p1_0 joined")
                p1_1.join()
                print("p1_1 joined")
                p2_0.join()
                print("p2_0 joined")
                p2_1.join()
                print("p2_1 joined")
                
                p3_0.start()
                print("p3_0 started")
                p3_1.start()
                print("p3_1 started")
                p4_0.start()
                print("p4_0 started")
                p4_1.start()
                print("p4_1 started")
                
                p3_0.join()
                print("p3_0 joined")
                p3_1.join()
                print("p3_1 joined")
                p4_0.join()
                print("p4_0 joined")
                p4_1.join()
                print("p4_1 joined")

                p5_0.start()
                print("p5_0 started")
                p5_1.start()
                print("p5_1 started")
                p6_0.start()
                print("p6_0 started")
                p6_1.start()
                print("p6_1 started")
                
                p5_0.join()
                print("p5_0 joined")
                p5_1.join()
                print("p5_1 joined")
                p6_0.join()
                print("p6_0 joined")
                p6_1.join()
                print("p6_1 joined")

                p7_0.start()
                print("p7_0 started")
                p7_1.start()
                print("p7_1 started")
                p8_0.start()
                print("p8_0 started")
                p8_1.start()
                print("p8_1 started")
                
                p7_0.join()
                print("p7_0 joined")
                p7_1.join()
                print("p7_1 joined")
                p8_0.join()
                print("p8_0 joined")
                p8_1.join()
                print("p8_1 joined")
                #shm.unlink()
                
                data_unshaped[sz:sz+int(sz/2)]=return_dict["1_0"]
                data_unshaped[sz+int(sz/2):sz+sz]=return_dict["1_1"]
                
                data_unshaped[sz*2:sz*2+int(sz/2)]=return_dict["2_0"]
                data_unshaped[sz*2+int(sz/2):2*sz+sz]=return_dict["2_1"]
                
                data_unshaped[sz*3:sz*3+int(sz/2)]=return_dict["3_0"]
                data_unshaped[sz*3+int(sz/2):3*sz+sz]=return_dict["3_1"]
                
                data_unshaped[sz*4:sz*4+int(sz/2)]=return_dict["4_0"]
                data_unshaped[sz*4+int(sz/2):4*sz+sz]=return_dict["4_1"]

                data_unshaped[sz*5:sz*5+int(sz/2)]=return_dict["5_0"]
                data_unshaped[sz*5+int(sz/2):5*sz+sz]=return_dict["5_1"]
                
                data_unshaped[sz*6:sz*6+int(sz/2)]=return_dict["6_0"]
                data_unshaped[sz*6+int(sz/2):6*sz+sz]=return_dict["6_1"]

                data_unshaped[sz*7:sz*7+int(sz/2)]=return_dict["7_0"]
                data_unshaped[sz*7+int(sz/2):7*sz+sz]=return_dict["7_1"]

                data_unshaped[sz*8:sz*8+int(sz/2)]=return_dict["8_0"]
                data_unshaped[sz*8+int(sz/2):8*sz+sz]=return_dict["8_1"]

                print(data_unshaped[sz+1])
                print(data_unshaped[sz*2+1])
                print(data_unshaped[sz*3+1])
                print(np.var(data_unshaped[sz*3+1]))
                print(data_unshaped[sz*3+2])
                print(data_unshaped[sz*3+3])
                print(np.var(data_unshaped[sz:2*sz]))
                print(np.var(data_unshaped[sz*2:3*sz]))
                print(np.var(data_unshaped[sz*3:4*sz]))
                print(np.var(data_unshaped[sz*4:5*sz]))
                self.mean=np.mean(data_unshaped)
                self.std=np.std(data_unshaped)
                return data_unshaped
        def batch_normalization(self,layer,index,size):
                y = tf.Variable(np.ones((size),np.float32))
                b = tf.Variable(np.zeros((size),np.float32))
                self.y_list.append(y)
                self.b_list.append(b)
                if len(self.batch_mean_plc_list)<=index:
                        self.batch_mean_plc_list.append(tf.compat.v1.placeholder(tf.float32,[size]))
                        self.batch_std_plc_list.append(tf.compat.v1.placeholder(tf.float32,[size]))                               
                graph = tf.compat.v1.get_default_graph()
                def f1(layer,graph,index):
                        with graph.as_default():
                                mean_x, std_x = tf.nn.moments(layer, axes = [0, 1, 2], keepdims=False)
                                if len(self.batch_mean_list)<=index:
                                        self.batch_mean_list.append(mean_x)
                                        self.batch_std_list.append(std_x)
                                else:
                                        self.batch_mean_list[index]=mean_x
                                        self.batch_std_list[index]=std_x
                                layer = tf.nn.batch_normalization(layer, mean_x, std_x, b, y, 1e-12)
                        return layer
                def f2(layer,graph,index):
                        with graph.as_default():
                                layer = tf.nn.batch_normalization(layer, self.batch_mean_plc_list[index], self.batch_std_plc_list[index], b, y, 1e-12)
                        return layer
                r = tf.cond(self.isTraining, lambda:f1(layer,graph,index), lambda:f2(layer,graph,index))
                return layer
        def model(self):
                self.data_place=tf.compat.v1.placeholder(tf.float32,[None,110250,2,1])
                self.y=tf.compat.v1.placeholder(tf.float32,[None,self.output_count])
                size=[5,5,1,8]
                std=initializer(size,"xavier_normal","relu")
                print("conv std:",std)
                kernel=tf.Variable(tf.compat.v1.random_normal(size, stddev=std))
                kernel1=kernel
                self.kernel1=kernel1
                bias=tf.Variable(tf.compat.v1.random_normal([size[3]],stddev=0.03))
                self.bias1=bias
                conv=tf.nn.conv2d(self.data_place, kernel, [1,1,1,1], "SAME")
                conv=tf.nn.bias_add(conv,bias)
                self.first1=conv
                conv = self.batch_normalization(conv,0,size[3])
                self.first2=conv
                conv=tf.nn.relu(conv)
                self.first3=conv
                conv=tf.nn.max_pool2d(conv,[1,16,1,1],[1,16,1,1],"SAME")
                self.kernels.append(kernel)
                self.kernels.append(bias)
                

                size=[3,3,8,16]
                std=initializer(size,"xavier_normal","relu") 
                kernel=tf.Variable(tf.compat.v1.random_normal(size, stddev=std))
                bias=tf.Variable(tf.compat.v1.random_normal([size[3]],stddev=0.03))
                self.bias2=bias
                self.kernel2=kernel
                conv=tf.nn.conv2d(conv, kernel, [1,1,1,1], "SAME")
                self.second1 = conv
                conv=tf.nn.bias_add(conv,bias)
                self.second2 = conv
                conv = self.batch_normalization(conv,1,size[3])
                self.second3 = conv
                conv=tf.nn.relu(conv)
                conv=tf.nn.max_pool2d(conv,[1,8,1,1],[1,8,1,1],"SAME")
                self.kernels.append(kernel)
                self.kernels.append(bias)
                
                

                size=[3,3,16,32]
                std=initializer(size,"xavier_normal","relu") 
                kernel=tf.Variable(tf.compat.v1.random_normal(size, stddev=std))
                self.kernel3=kernel
                bias=tf.Variable(tf.compat.v1.random_normal([size[3]],stddev=0.03))
                self.bias3=bias
                conv=tf.nn.conv2d(conv, kernel, [1,1,1,1], "SAME")
                conv=tf.nn.bias_add(conv,bias)
                conv = self.batch_normalization(conv,2,size[3])
                conv=tf.nn.relu(conv)
                conv=tf.nn.max_pool2d(conv,[1,4,1,1],[1,4,1,1],"SAME")
                self.kernels.append(kernel)
                self.kernels.append(bias)

                size=[3,3,32,64]
                std=initializer(size,"xavier_normal","relu") 
                kernel=tf.Variable(tf.compat.v1.random_normal(size, stddev=std))
                self.kernel4=kernel
                bias=tf.Variable(tf.compat.v1.random_normal([size[3]],stddev=0.03))
                self.bias4=bias
                conv=tf.nn.conv2d(conv, kernel, [1,1,1,1], "SAME")
                conv=tf.nn.bias_add(conv,bias)
                conv = self.batch_normalization(conv,3,size[3])
                conv=tf.nn.relu(conv)
                conv=tf.nn.max_pool2d(conv,[1,2,1,1],[1,2,1,1],"SAME")
                self.kernels.append(kernel)
                self.kernels.append(bias)
                

                size=[3,3,64,96]
                std=initializer(size,"xavier_normal","relu") 
                kernel=tf.Variable(tf.compat.v1.random_normal(size, stddev=std))
                self.kernel5=kernel
                bias=tf.Variable(tf.compat.v1.random_normal([size[3]],stddev=0.03))
                self.bias5=bias
                conv=tf.nn.conv2d(conv, kernel, [1,1,1,1], "SAME")
                conv=tf.nn.bias_add(conv,bias)
                self.fifth1=conv
                conv = self.batch_normalization(conv,4,size[3])
                self.fifth2=conv
                conv=tf.nn.relu(conv)
                self.fifth3=conv
                conv=tf.nn.max_pool2d(conv,[1,2,1,1],[1,2,1,1],"SAME")
                self.kernels.append(kernel)
                self.kernels.append(bias)
                self.keep_prob2=tf.compat.v1.placeholder(tf.float32,[])
                conv=tf.nn.dropout(conv,self.keep_prob2)

                size=[3,3,96,128]
                std=initializer(size,"xavier_normal","relu") 
                kernel=tf.Variable(tf.compat.v1.random_normal(size, stddev=std))
                self.kernel6=kernel
                bias=tf.Variable(tf.compat.v1.random_normal([size[3]],stddev=0.03))
                self.bias6=bias
                conv=tf.nn.conv2d(conv, kernel, [1,1,1,1], "SAME")
                conv=tf.nn.bias_add(conv,bias)
                self.last = conv
                conv = self.batch_normalization(conv,5,size[3])
                self.last2=conv
                conv=tf.nn.relu(conv)
                self.last3=conv
                conv=tf.nn.max_pool2d(conv,[1,2,1,1],[1,2,1,1],"SAME")
                self.kernels.append(kernel)
                self.kernels.append(bias)
                conv=tf.nn.dropout(conv,self.keep_prob2)

                size=[3,3,128,256]
                std=initializer(size,"xavier_normal","relu") 
                kernel=tf.Variable(tf.compat.v1.random_normal(size, stddev=std))
                self.kernel7=kernel
                bias=tf.Variable(tf.compat.v1.random_normal([size[3]],stddev=0.03))
                self.bias7=bias
                conv=tf.nn.conv2d(conv, kernel, [1,1,1,1], "SAME")
                conv=tf.nn.bias_add(conv,bias)
                self.last = conv
                conv = self.batch_normalization(conv,6,size[3])
                self.last2=conv
                conv=tf.nn.relu(conv)
                self.last3=conv
                conv=tf.nn.max_pool2d(conv,[1,2,1,1],[1,2,1,1],"SAME")
                self.kernels.append(kernel)
                self.kernels.append(bias)
                conv=tf.nn.dropout(conv,self.keep_prob2)
                
                size=[3,3,256,512]
                std=initializer(size,"xavier_normal","relu") 
                kernel=tf.Variable(tf.compat.v1.random_normal(size, stddev=std))
                self.kernel8=kernel
                bias=tf.Variable(tf.compat.v1.random_normal([size[3]],stddev=0.03))
                self.bias8=bias
                conv=tf.nn.conv2d(conv, kernel, [1,1,1,1], "SAME")
                conv=tf.nn.bias_add(conv,bias)
                self.last = conv
                conv = self.batch_normalization(conv,7,size[3])
                self.last2=conv
                conv=tf.nn.relu(conv)
                self.last3=conv
                conv=tf.nn.max_pool2d(conv,[1,2,1,1],[1,2,1,1],"SAME")
                self.kernels.append(kernel)
                self.kernels.append(bias)
                conv=tf.nn.dropout(conv,self.keep_prob2)
                

                

                input_count=1
                shapes=conv.shape
                print(shapes)
                for shape in shapes:
                    if shape!=None:
                        input_count*=shape
                print(input_count)
                flat=tf.reshape(conv,[-1,input_count])
                self.regularization=0
                input_count=int(input_count)
                kernel_sizes=[input_count, int(input_count/5*3)]
                std=initializer(kernel_sizes,"xavier_normal","relu")
                print("init std:",std)
                W = tf.Variable(tf.compat.v1.random_normal([input_count, int(input_count/5*3)], stddev=std))
                self.W1=W
                b = tf.Variable(tf.compat.v1.random_normal([int(input_count/5*3)],stddev=0.01))
                hidden_out = tf.add(tf.matmul(flat, W), b)
                hidden_out=tf.nn.relu(hidden_out)
                self.kernels.append(W)
                self.kernels.append(b)
                self.w1_reg=0.00003
                self.regularization=self.regularization+self.w1_reg*tf.reduce_sum(tf.square(W))+tf.reduce_sum(tf.square(b))
                                        
                self.keep_prob=tf.compat.v1.placeholder(tf.float32,[])
                hidden_out=tf.nn.dropout(hidden_out,self.keep_prob)
                input_count=int(input_count/5*3)
                

                output_count=self.output_count
                kernel_sizes=[input_count, output_count]
                std=initializer(kernel_sizes,"xavier_normal","softmax")
                W = tf.Variable(tf.compat.v1.random_normal([input_count, output_count], stddev=std))
                self.W3=W
                b = tf.Variable(tf.compat.v1.random_normal([output_count],stddev=0.01))
                hidden_out = tf.add(tf.matmul(hidden_out, W), b)
                hidden_out=tf.nn.sigmoid(hidden_out)
                self.kernels.append(W)
                self.kernels.append(b)
                self.w2_reg=0.0008
                self.regularization=self.regularization+self.w2_reg*tf.reduce_sum(tf.square(W))+tf.reduce_sum(tf.square(b))
                self.hidden_out = hidden_out

                #liste_test=[]
                #for q in range(12):
                #        liste_test+=liste_test1
                self.loss=loss_function(hidden_out,self.y,loss_type="binary")
                self.lr=tf.compat.v1.placeholder(tf.float32,[])
                self.alpha=0.00015
                self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss+self.regularization)

                self.correct_prediction = tf.equal(tf.argmax(hidden_out, 1),tf.argmax(self.y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

                self.first_mean1=tf.math.reduce_mean(self.first1)
                self.first_var1=tf.math.reduce_variance(self.first1)
                self.first_mean2=tf.math.reduce_mean(self.first2)
                self.first_var2=tf.math.reduce_variance(self.first2)
                self.first_mean3=tf.math.reduce_mean(self.first3)
                self.first_var3=tf.math.reduce_variance(self.first3)

                self.second_mean1=tf.math.reduce_mean(self.second1)
                self.second_var1=tf.math.reduce_variance(self.second1)
                self.second_mean2=tf.math.reduce_mean(self.second2)
                self.second_var2=tf.math.reduce_variance(self.second2)
                self.second_mean3=tf.math.reduce_mean(self.second3)
                self.second_var3=tf.math.reduce_variance(self.second3)
                
                self.fifth_mean1=tf.math.reduce_mean(self.fifth1)
                self.fifth_var1=tf.math.reduce_variance(self.fifth1)
                self.fifth_mean2=tf.math.reduce_mean(self.fifth2)
                self.fifth_var2=tf.math.reduce_variance(self.fifth2)
                self.fifth_mean3=tf.math.reduce_mean(self.fifth3)
                self.fifth_var3=tf.math.reduce_variance(self.fifth3)
                
                self.last_mean=tf.math.reduce_mean(self.last)
                self.last_var=tf.math.reduce_variance(self.last)
                self.last_mean2=tf.math.reduce_mean(self.last2)
                self.last_var2=tf.math.reduce_variance(self.last2)
                self.last_mean3=tf.math.reduce_mean(self.last3)
                self.last_var3=tf.math.reduce_variance(self.last3)

                
                init = tf.compat.v1.global_variables_initializer()
                self.sess=tf.compat.v1.Session()
                self.sess.run(init)
        def keras_model(self):
                self.graph = tf.compat.v1.get_default_graph()
                leaky_alpha=0.01
                with self.graph.as_default():
                        self.model = keras.Sequential()
                        self.model.add(layers.Conv2D(16, (3, 3), input_shape=(110250,2,1),padding="SAME"))
                        self.model.add(layers.LeakyReLU(alpha=leaky_alpha))
                        self.model.add(layers.MaxPooling2D((32, 32),padding="SAME"))
                        self.model.add(layers.BatchNormalization())
                        self.model.add(layers.Conv2D(32, (3, 3),padding="SAME"))
                        self.model.add(layers.LeakyReLU(alpha=leaky_alpha))
                        self.model.add(layers.MaxPooling2D((16, 16),padding="SAME"))
                        self.model.add(layers.BatchNormalization())
                        self.model.add(layers.Conv2D(64, (3, 3),padding="SAME"))
                        self.model.add(layers.LeakyReLU(alpha=leaky_alpha))
                        self.model.add(layers.MaxPooling2D((4, 4),padding="SAME"))
                        self.model.add(layers.BatchNormalization())
                        self.model.add(layers.Conv2D(96, (3, 3),padding="SAME"))
                        self.model.add(layers.LeakyReLU(alpha=leaky_alpha))
                        self.model.add(layers.MaxPooling2D((2, 2),padding="SAME"))
                        self.model.add(layers.BatchNormalization())
                        self.model.add(layers.Conv2D(128, (3, 3),padding="SAME"))
                        self.model.add(layers.LeakyReLU(alpha=leaky_alpha))
                        self.model.add(layers.MaxPooling2D((2, 2),padding="SAME"))
                        self.model.add(layers.BatchNormalization())
                        self.model.add(layers.Conv2D(256, (3, 3),padding="SAME"))
                        self.model.add(layers.LeakyReLU(alpha=leaky_alpha))
                        self.model.add(layers.MaxPooling2D((2, 2),padding="SAME"))
                        self.model.add(layers.BatchNormalization())
                        self.model.add(layers.Flatten())
                        self.model.add(layers.Dense(1100,kernel_regularizer=regularizers.l1_l2(l1=3e-5, l2=3e-4),
                                    bias_regularizer=regularizers.l2(1e-4),
                                    activity_regularizer=regularizers.l2(1e-5)))
                        self.model.add(layers.LeakyReLU(alpha=leaky_alpha))
                        self.model.add(keras.layers.Dropout(rate=0.5))
                        self.model.add(layers.Dense(self.output_count,kernel_regularizer=regularizers.l1_l2(l1=3e-5, l2=3e-4),
                                bias_regularizer=regularizers.l2(1e-4),
                                activity_regularizer=regularizers.l2(1e-5)))
                        self.model.add(layers.Activation(activations.softmax))
                        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                                loss=tf.keras.losses.CategoricalCrossentropy(),#my_loss(self.output_count/2),
                                metrics=[tf.keras.metrics.BinaryAccuracy(),
                                       tf.keras.metrics.FalseNegatives(thresholds=0.2),tf.keras.metrics.FalsePositives(thresholds=0.2)])
                        
                        self.sess=tf.compat.v1.Session()
                        #session = keras.backend.get_session()
                        init = tf.compat.v1.global_variables_initializer()
                        self.sess.run(init)
                
        def add_plc(self,feed,batch_mean_list,batch_std_list,batch_count):
                for i in range(len(batch_mean_list)):
                                
                        feed[self.batch_mean_plc_list[i]]=batch_mean_list[i]/batch_count
                        feed[self.batch_std_plc_list[i]]=batch_std_list[i]/batch_count
                return feed
        def create_list(self,arr):
                liste = []
                for i in arr:
                        arr = np.array(list(range(i)),np.float64)
                        arr.fill(0)
                        liste.append(np.copy(arr))
                return liste
        def start_training(self,data_unshaped_main,data_unshaped):
                if self.control_window_bool==True:
                        self.control_window()
                        self.train_thread=Thread(target=lambda data_unshaped_main,data_unshaped,graph,sess: self.keras_train(data_unshaped_main,data_unshaped,graph,sess),
                                                 args=(data_unshaped_main,data_unshaped,self.graph,self.sess))
                        self.train_thread.start()
                        self.tk.mainloop()
                else:
                        self.train(data_unshaped_main,data_unshaped)
        def keras_train(self,data_unshaped_main,data_unshaped,graph,sess):
                with graph.as_default():
                        self.model.summary()
                        
                        checkpoint_filepath = './model'
                        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                         save_weights_only=False,save_best_only=True,
                                                         verbose=1)
                        tekrar=9
                        set_session(sess)

                        
                        #data_unshaped_main=self.get_data()
                        print(data_unshaped_main)
                        
                        sz=len(data_unshaped_main)
                        print(sz)
                        sayac=sz
                        #data_unshaped=np.zeros((sz*tekrar,110250,2),np.float32)
                        data=np.zeros((sz*tekrar,110250,2 , 1),np.float32)
                        one_hot=[]
                        for i in range(tekrar):
                                one_hot+=self.one_hot_main
                        one_hot=np.array(one_hot,np.float32)
                        print("one hot:",one_hot)
                        data_unshaped[:sz]=np.copy(data_unshaped_main)
                        data= np.reshape(data_unshaped,(sz*tekrar,110250,2 , 1))
                        
                        

                        #norm= np.linalg.norm(data)
                        #data = data/norm
                        #data=np.copy(data[:36])
                        #one_hot=np.copy(one_hot[:36])
                        std=np.std(data)
                        mean=np.mean(data)
                        mean_var = open("model/mean_var.txt","w")
                        
                        self.std=std
                        self.mean=mean
                        print(str(self.mean)+" "+str(self.std))
                        mean_var.write(str(self.mean)+" "+str(self.std))
                        mean_var.close()

                        sz_test=len(os.listdir("./test/"))
                        #data_unshaped_test=np.zeros((sz_test,110250,2),np.float32)
                        test_sayac=0
                        one_hot_test=self.one_hot_test
                        """test_indexes=list(range(0,sz_test))
                        print("test indexes:",test_indexes)
                        for i in range(0,sz_test):
                                
                                ng, fs=  sf.read("./test/record_"+str(i)+".wav", dtype='float32')
                                data_unshaped_test[test_sayac]=ng
                                one_hot_test.append(self.fill_one_hot(test_indexes[test_sayac]))
                                test_sayac+=1
                        data_test= np.reshape(data_unshaped_test,(sz_test,110250,2 , 1))"""
                        data_test= np.reshape(self.test_data,(self.test_data.shape[0],110250,2 , 1))
                        one_hot_test=np.array(one_hot_test)
                        print("test one_hot",one_hot_test)
                        print("test önce:",np.mean(data_test),np.var(data_test))
                        print("data önce:",np.mean(data),np.var(data))
                        
                        print(np.mean(data),np.var(data))
                        print(0,sz,np.mean(data[0:sz]),np.var(data[:sz]))
                        print(sz,sz*2,np.mean(data[sz:sz*2]),np.var(data[sz:sz*2]))
                        print(sz*2,sz*3,np.mean(data[sz*2:sz*3]),np.var(data[sz*2:sz*3]))
                        print(sz*3,sz*4,np.mean(data[sz*3:sz*4]),np.var(data[sz*3:sz*4]))
                        print(sz*4,sz*5,np.mean(data[sz*4:sz*5]),np.var(data[sz*4:sz*5]))
                        data=(data-mean)/std
                        data_test=(data_test-mean)/std
                        print(np.mean(data_test))
                        print(np.var(data_test))
                        """mean1=tf.math.reduce_mean(self.W1)
                        std1=tf.math.reduce_variance(self.W1)
                        #mean2=tf.math.reduce_mean(self.W2)
                        #std2=tf.math.reduce_variance(self.W2)
                        mean3=tf.math.reduce_mean(self.W3)
                        std3=tf.math.reduce_variance(self.W3)"""
                        
                        batch_size=50
                        limit=500
                         
                        data,one_hot=shuffling(data,one_hot)
                        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10, min_lr=1e-4)
                            
                        self.model.fit(data, 
                          one_hot,
                          batch_size=batch_size,
                          epochs=limit,
                          validation_data=(data_test, one_hot_test),
                          callbacks=[self.cp_callback ,reduce_lr])  # Pass callback to training
                        self.model.save("model2/cp.h5")
                        print("geldi")
                        self.tk.destroy()

        def train(self,data_unshaped_main,data_unshaped,graph):
                with graph.as_default():
                        tekrar=5
                        

                        
                        #data_unshaped_main=self.get_data()
                        print(data_unshaped_main)
                        
                        sz=len(data_unshaped_main)
                        print(sz)
                        sayac=sz
                        #data_unshaped=np.zeros((sz*tekrar,110250,2),np.float32)
                        data=np.zeros((sz*tekrar,110250,2 , 1),np.float32)
                        one_hot=[]
                        for i in range(tekrar):
                                one_hot+=self.one_hot_main
                        one_hot=np.array(one_hot,np.float32)
                        print("one hot:",one_hot)
                        data_unshaped[:sz]=np.copy(data_unshaped_main)
                        
                        

                        data= np.reshape(data_unshaped,(sz*tekrar,110250,2 , 1))

                        

                        #norm= np.linalg.norm(data)
                        #data = data/norm
                        #data=np.copy(data[:36])
                        #one_hot=np.copy(one_hot[:36])
                        std=np.std(data)
                        mean=np.mean(data)
                        mean_var = open("model/mean_var.txt","w")
                        
                        self.std=std
                        self.mean=mean
                        print(str(self.mean)+" "+str(self.std))
                        mean_var.write(str(self.mean)+" "+str(self.std))
                        mean_var.close()

                        sz_test=len(os.listdir("./test/"))
                        data_unshaped_test=np.zeros((sz_test,110250,2),np.float32)
                        test_sayac=0
                        one_hot_test=[]
                        test_indexes=list(range(0,sz_test))
                        print("test indexes:",test_indexes)
                        for i in range(0,sz_test):
                                
                                ng, fs=  sf.read("./test/record_"+str(i)+".wav", dtype='float32')
                                data_unshaped_test[test_sayac]=ng
                                one_hot_test.append(self.fill_one_hot(test_indexes[test_sayac]))
                                test_sayac+=1
                        data_test= np.reshape(data_unshaped_test,(sz_test,110250,2 , 1))
                        one_hot_test=np.array(one_hot_test)
                        print("test one_hot",one_hot_test)
                        print("test önce:",np.mean(data_test),np.var(data_test))
                        print("data önce:",np.mean(data),np.var(data))
                        
                        print(np.mean(data),np.var(data))
                        print(0,sz,np.mean(data[0:sz]),np.var(data[:sz]))
                        print(sz,sz*2,np.mean(data[sz:sz*2]),np.var(data[sz:sz*2]))
                        print(sz*2,sz*3,np.mean(data[sz*2:sz*3]),np.var(data[sz*2:sz*3]))
                        print(sz*3,sz*4,np.mean(data[sz*3:sz*4]),np.var(data[sz*3:sz*4]))
                        print(sz*4,sz*5,np.mean(data[sz*4:sz*5]),np.var(data[sz*4:sz*5]))
                        data=(data-mean)/std
                        data_test=(data_test-mean)/std
                        print(np.mean(data_test))
                        print(np.var(data_test))
                        mean1=tf.math.reduce_mean(self.W1)
                        std1=tf.math.reduce_variance(self.W1)
                        #mean2=tf.math.reduce_mean(self.W2)
                        #std2=tf.math.reduce_variance(self.W2)
                        mean3=tf.math.reduce_mean(self.W3)
                        std3=tf.math.reduce_variance(self.W3)
                        
                        print("init bitti")
                        batch_size=137
                        limit=120
                        ep=0
                        self.initial_learning_rate=0.001
                        self.learning_rate=self.initial_learning_rate
                        last_acc=0
                        self.aug_tresh=20
                        self.train_acc_list=[]
                        self.train_loss_list=[]
                        self.acc_list=[]
                        self.loss_list=[]
                        last_aug=0
                        self.aug_input=False
                        self.keep1=0.45
                        self.keep2=0.05
                        while ep<limit:
                                print("kernel mean and vars:")
                                kern1=self.sess.run(self.kernel1)
                                kern2=self.sess.run(self.kernel2)
                                kern3=self.sess.run(self.kernel3)
                                kern4=self.sess.run(self.kernel4)
                                kern5=self.sess.run(self.kernel5)
                                kern6=self.sess.run(self.kernel6)
                                kern7=self.sess.run(self.kernel7)
                                kern8=self.sess.run(self.kernel8)
                                print("kernel 1 mean var:",np.mean(kern1),np.var(kern1))
                                print("kernel 2 mean var:",np.mean(kern2),np.var(kern2))
                                print("kernel 3 mean var:",np.mean(kern3),np.var(kern3))
                                print("kernel 4 mean var:",np.mean(kern4),np.var(kern4))
                                print("kernel 5 mean var:",np.mean(kern5),np.var(kern5))
                                print("kernel 6 mean var:",np.mean(kern6),np.var(kern6))
                                print("kernel 7 mean var:",np.mean(kern7),np.var(kern8))
                                print("kernel 8 mean var:",np.mean(kern6),np.var(kern6))
                                print("bias mean and vars:")
                                bias1=self.sess.run(self.bias1)
                                bias2=self.sess.run(self.bias2)
                                bias3=self.sess.run(self.bias3)
                                bias4=self.sess.run(self.bias4)
                                bias5=self.sess.run(self.bias5)
                                bias6=self.sess.run(self.bias6)
                                bias7=self.sess.run(self.bias7)
                                bias8=self.sess.run(self.bias8)
                                print("bias 1 mean var:",np.mean(bias1),np.var(bias1))
                                print("bias 2 mean var:",np.mean(bias2),np.var(bias2))
                                print("bias 3 mean var:",np.mean(bias3),np.var(bias3))
                                print("bias 4 mean var:",np.mean(bias4),np.var(bias4))
                                print("bias 5 mean var:",np.mean(bias5),np.var(bias5))
                                print("bias 6 mean var:",np.mean(bias6),np.var(bias6))
                                print("bias 7 mean var:",np.mean(bias7),np.var(bias7))
                                print("bias 8 mean var:",np.mean(bias8),np.var(bias8))
                                batch_mean_list=self.create_list([8,16,32,64,96,128])
                                batch_std_list=self.create_list([8,16,32,64,96,128])
                                data,one_hot=shuffling(data,one_hot)
        ##                        print("y and b:")
        ##                        print(self.sess.run(self.y_list))
        ##                        print(self.sess.run(self.b_list))
                                print("epoch:",ep+1)
                                total_kayıp=0
                                total_acc=0
                                for part in range(int(len(data)/batch_size)):
                                       
                                       batch=data[part*batch_size:(part+1)*batch_size]
                                       batch_ground=one_hot[part*batch_size:(part+1)*batch_size]
                                       """feed = {self.data_place:batch,self.y:batch_ground,self.keep_prob:0,self.keep_prob2:0,
                                                        self.isTraining:True}
                                       feed = self.add_plc(feed,batch_mean_list,batch_std_list,int(len(data)/batch_size))
                                       batch_layer_mean = self.sess.run(self.batch_mean_list,feed_dict=feed)
                                       batch_layer_std = self.sess.run(self.batch_std_list,feed_dict=feed)
                                       for i in range(len(batch_layer_mean)):
                                               if len(batch_mean_list)<=i:
                                                     batch_mean_list.append(batch_layer_mean[i])
                                                     batch_std_list.append(batch_layer_std[i])
                                               else:
                                                     batch_mean_list[i]+=batch_layer_mean[i]
                                                     batch_std_list[i]+=batch_layer_std[i]"""
                                               
                                       #last_batch_mean=np.mean(np.array(batch_mean_list),axis=0)
                                       #last_batch_std=np.std(np.array(batch_std_list),axis=0)
                                       if part==0 or part==1:
                                               print("batch mean and var:",np.mean(batch),np.var(batch))
                                               feed={self.data_place:batch,self.y:batch_ground,self.keep_prob:0,self.keep_prob2:0,
                                                        self.isTraining:True}
                                               feed = self.add_plc(feed,batch_mean_list,batch_std_list,int(len(data)/batch_size))
                                               bat=self.sess.run([self.first_mean1,self.first_var1,
                                                                  self.first_mean2,self.first_var2,
                                                                  self.first_mean3,self.first_var3,
                                                                  self.second_mean1,self.second_var1,
                                                                  self.second_mean2,self.second_var2,
                                                                  self.second_mean3,self.second_var3,
                                                                  self.fifth_mean1,self.fifth_var1,
                                                                  self.fifth_mean2,self.fifth_var2,
                                                                  self.fifth_mean3,self.fifth_var3,self.last_mean,self.last_var,
                                                                  self.last_mean2,self.last_var2,self.last_mean3,self.last_var3,self.batch_mean_list[0],self.batch_std_list[0]],
                                                                 feed_dict=feed)
                                               print("means and vars:",bat)
                                       feed = {self.data_place:batch,self.y:batch_ground,self.keep_prob:self.keep1,self.keep_prob2:self.keep2,self.lr:self.learning_rate,
                                                        self.isTraining:True}
                                       feed = self.add_plc(feed,batch_mean_list,batch_std_list,int(len(data)/batch_size))
                                       kayıp,_,train_batch_acc=self.sess.run([self.loss,self.opt,self.accuracy],feed_dict=feed)
                                       total_kayıp+=kayıp
                                       total_acc+=train_batch_acc
                                print("dataset total loss:",total_kayıp/int(len(data)/batch_size))
                                print("dataset total accuracy:",total_acc/int(len(data)/batch_size))
                                self.train_acc_list.append(total_acc/int(len(data)/batch_size))
                                self.train_loss_list.append(total_kayıp/int(len(data)/batch_size))
                                ds_kayıp=total_kayıp/int(len(data)/batch_size)
                                feed={self.data_place:data_test,self.y:one_hot_test,self.keep_prob:0,
                                        self.keep_prob2:0,
                                        self.isTraining:False}
                                feed = self.add_plc(feed,batch_mean_list,batch_std_list,int(len(data)/batch_size))
                                kayıp=self.sess.run([self.loss,self.accuracy,self.hidden_out,self.correct_prediction],feed_dict=feed)
                                acc=kayıp[1]
                                self.acc_list.append(acc)
                                self.loss_list.append(kayıp[0])
                                print("acc list:",self.acc_list)
                                print("loss list:",self.loss_list)
                                print("test:",kayıp[0],kayıp[1],kayıp[3],kayıp[2][0],kayıp[2][20])
                                mn=self.sess.run([mean1,std1])
                                print("mean std:",mn)
                                #mn=self.sess.run([mean2,std2])
                                #print("mean std2:",mn)
                                mn=self.sess.run([mean3,std3])
                                print("mean std3:",mn)
                                #w_2=self.sess.run(self.W2)
                                #print("w2:",w_2[:100])
                                ep+=1
                                self.learning_rate = self.initial_learning_rate * 1/(1 + 0.1 * ep)
                                print("new learning rate:",self.learning_rate)
                                if acc>0.65 and acc>last_acc:
                                        saver = tf.compat.v1.train.Saver(self.kernels)
                                        save_path = saver.save(self.sess,"./model/"+"model.ckpt",global_step=1)
                                        last_acc=acc
                                if ep==limit or kayıp[0]<1 or self.stop==True:
                                        
                                        print("devam edilsin mi?")
                                        sa=input()
                                        if sa!="e":
                                                try:
                                                        limit+=int(sa)
                                                except:
                                                        limit+=10
                                        else:
                                                saver = tf.compat.v1.train.Saver(self.kernels)
                                                save_path = saver.save(self.sess,"./model/"+"model.ckpt",global_step=1)
                                if (ep-last_aug>=self.aug_tresh and ep!=0 and ds_kayıp<0.8) or self.aug_input==True :
                                        print("augmentation başladı!")
                                        data_unshaped=self.augmentation()
                                        data_unshaped[:sz]=np.copy(data_unshaped_main)
                                        data= np.reshape(data_unshaped,(sz*tekrar,110250,2 , 1))
                                        std=np.std(data)
                                        mean=np.mean(data)
                                        mean_var = open("model/mean_var.txt","w")
                                        self.std=std
                                        self.mean=mean
                                        print(str(self.mean)+" "+str(self.std))
                                        mean_var.write(str(self.mean)+" "+str(self.std))
                                        mean_var.close()
                                        data_test= np.reshape(data_unshaped_test,(sz_test,110250,2 , 1))
                                        data=(data-mean)/std
                                        data_test=(data_test-mean)/std
                                        one_hot=[]
                                        for i in range(tekrar):
                                                one_hot+=self.one_hot_main
                                        one_hot=np.array(one_hot,np.float32)
                                        print("one hot:",one_hot)
                                        last_aug=ep
                
                                        
        def search(self,index):
                dosya = open ("belgeler/sozluk.txt","r",encoding="UTF-8")
                lines = dosya.readlines()
                part1 = index.split("_")[0]
                part2 = index.split("_")[1]
                line = lines[int(part1)]
                line_parts = line.split("-")
                if self.source=="tr_voices":
                        words = line_parts[2].split(",")
                elif self.source=="eng_voices":
                        words = line_parts[1].split(",")
                return words[int(part2)]
        def load(self):
                try:
                        saver = tf.compat.v1.train.Saver(self.kernels)
                        saver.restore(self.sess,tf.train.latest_checkpoint("./model"))
                        mean_var = open("model/mean_var.txt","r")
                        lines =mean_var.readlines() 
                        print(lines[0])
                        print(lines[0].split(" "))
                        self.mean= float(lines[0].split(" ")[0])
                        self.std= float(lines[0].split(" ")[1])
                        mean_var.close()
                except:
                        None
        def dinle(self,arr):
            if arr.shape==(110250,2):
                    data_unshaped_test=np.zeros((1,110250,2),np.float32)
                    data_unshaped_test[0]=arr
                    data_test= np.reshape(data_unshaped_test,(1,110250,2 , 1))
            else:
                    data_test= np.reshape(arr,(arr.shape[0],110250,2 , 1))
            
            data_test=(data_test-self.mean)/self.std
            liste = self.sess.run(self.hidden_out,feed_dict={self.data_place:data_test,self.keep_prob:0,self.keep_prob2:0})[0]
            result={}
            index = 0 
            for i in liste:
                    if i>0.1:
                            result[self.search(self.dictionary[index])] = i
                    index+=1
            return result
        def keras_load(self):
            self.get_data(False)
            #self.keras_model()
            self.graph=tf.compat.v1.get_default_graph()
            self.sess=tf.compat.v1.Session()
            set_session(self.sess)
            weighted_cross_entropy_with_logits=my_loss(self.output_count/2)
            self.model=keras.models.load_model("model/", custom_objects={'weighted_cross_entropy_with_logits': weighted_cross_entropy_with_logits})
            mean_var = open("model/mean_var.txt","r")
            lines =mean_var.readlines() 
            print(lines[0])
            print(lines[0].split(" "))
            self.mean= float(lines[0].split(" ")[0])
            self.std= float(lines[0].split(" ")[1])
            mean_var.close()
            self.a=0
        def rnd_aug(self,arr):
                rnd = random.randint(0,3)
                if rnd==0:
                        aug = complex_aug(arr)
                elif rnd==1:
                        aug = complex_aug2(arr)
                elif rnd==2:
                        aug = complex_aug3(arr)
                elif rnd==3:
                        aug = complex_aug4(arr)
                return aug
        def keras_predict(self,arr):
                """if self.a==0:
                        self.yed=arr[:]
                        self.a+=1
                elif self.a==1:
                        arr[:]=self.yed[:]
                        self.model.load_weights("model2/cp.h5")"""
                        
                with self.sess.as_default():
                        with self.graph.as_default():
                                #set_session(self.sess)
                                if arr.shape==(110250,2):
                                    tekrar=3
                                            
                                    data_unshaped_test=np.zeros((tekrar,110250,2),np.float32)
                                    data_unshaped_test[0]=arr
                                    for i in range(1,tekrar):
                                            data_unshaped_test[i]=self.rnd_aug(arr)
                                    #data_unshaped_test[3]=aug3
                                    #data_unshaped_test[4]=aug4
                                    data_test= np.reshape(data_unshaped_test,(data_unshaped_test.shape[0],110250,2 , 1))
                                else:
                                    data_test= np.reshape(arr,(arr.shape[0],110250,2 , 1))
                                data_test=(data_test-self.mean)/self.std
                                pred=self.model.predict(data_test)
                                print("pred:",pred)
                                #pred= sigmoid(pred)
                                #print(pred)
                                acc = (pred > 0.2).astype("int32")
                                #print(self.sess.run(pred))
                result={}
                
                for q in acc:
                        index = 0
                        for i in q:
                                if i>0.1:
                                        res = self.search(self.dictionary[index])
                                        if res not in result:
                                                result[res] = i
                                        else:
                                                for i in range(1000):
                                                        if str(res)+"-"+str(i) not in result:
                                                                result[str(res)+"-"+str(i)] = i
                                                                break
                                index+=1
                return result
        def listen_console():
                while True:
                        for line in sys.stdin:
                                if line.split(":")[0]=="input" and len(line.split(":"))>1 :
                                        code=str(line.split(":")[1])
                                        x=False
                                        x = re.search("import", txt)
                                        x = re.search("sys", txt)
                                        x = re.search("os/.", txt)
                                        x = re.search("Process", txt)
                                        x = re.search("Array", txt)
                                        x = re.search("multiprocessing", txt)
                                        if x==False:
                                                eval(code)
                                if self.print_flag==True:
                                        self.print_flag=False
                                        break
                        time.sleep(5)
                        




