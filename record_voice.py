# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import speech_ai
import soundfile as sf
global sayac
from tkinter import *
import os
import numpy as np
import threading
import random
def listen(duration=15,freq=44100):
    recording = sd.rec(int(duration * freq), 
                       samplerate=freq, channels=2)
    sd.wait()
    return recording
def save_record(index,index2=0,record_index=0,data=None):
    freq=44100
    recording=data
    if not os.path.exists("./eng_voices/"+str(index)+"_"+str(index2)):
        os.makedirs("./eng_voices/"+str(index)+"_"+str(index2))
        
    write("./eng_voices/"+str(index)+"_"+str(index2)+"/record_"+str(record_index)+".wav", freq, recording)
    return
def parçala(data,part=6):
    data_list=[]
    length=len(data)
    part_len=int(length/part)
    for i in range(part):
        data_list.append(np.copy(data[part_len*i:part_len*(i+1)]))
    return data_list
def save_test():
    freq=44100
    sayac=137
    while True:
        print("kaydetmek için bir tuşa bas:")
        input()
        recording=listen(duration=2.5)
        write("./test/"+"record_"+str(sayac)+".wav", freq, recording)
        sayac+=1
class listener:
    def __init__(self):
        self.s2=0
        self.s=0
        self.tr=speech_ai.training()
        self.flag=False
        self.answer=None
        self.answer_flag=False
        self.first=True
        self.sayac=0
        self.voice_size=110250
        self.voice_sec=2.5
        self.test_arr = np.zeros((1,self.voice_size,2),np.float32)
        self.nw=0.25
        self.part_size=int(self.voice_size/self.voice_sec*self.nw)
        self.test_arr = np.zeros((1,self.voice_size,2),np.float32)
    def live(self):
        
        """self.test_size=5

        
        sayac=0
        first=True
        print("started to listen")
        while True:
            if first==True:
                self.last_record = listen(self.voice_sec)
                first=False
            else:
                self.last_record = listen(self.nw)
            self.flag=True"""
        block_size = int(44100 *self.nw)
        self._input_stream = sd.InputStream( samplerate=44100,device=1, dtype='float32', callback=lambda indata, frames, time, status:self.test(indata,frames,time,status),channels=2,latency=0,
                                            blocksize=block_size)
        self._input_stream.start()
                    
            
    def define_window(self):
        dosya=open("belgeler/sozluk.txt","r",encoding="utf-8")
        self.lines=dosya.readlines()
        self.index=389
        self.index2=0
        self.record_index=-1
        self.pencere=Tk()
        self.label=Label(text=self.lines[self.index].split("-")[1].split(",")[self.index2])
        self.label.pack()
        self.buton=Button(text="listen",command=lambda :self.new_word())
        self.buton.focus_force()
        self.buton.pack()
        self.label2=Label(text="record:"+str(self.record_index))
        self.label2.pack()
        self.label3=Label(text="")
        self.label3.pack()
        #self.save_model()
        #self.finish()
        self.pencere.mainloop()
    def new_word(self):
        self.buton.focus_force()
        line=self.lines[self.index]
        parts=line.split("-")
        turkce=parts[1]
        print(turkce)
        turkce_parts=turkce.split(",")
        
        print(self.index)
        print(self.index2)
        self.label["text"]=turkce_parts[self.index2]
        self.label2["text"]="record:"+str(self.record_index)
        self.label3["text"]="dinliyor"
        self.pencere.update_idletasks()
        data_parts=parçala(listen(5),2)
        for i in range(len(data_parts)):
            save_record(self.index,self.index2,i,np.copy(data_parts[i]))
        self.label3["text"]="bitti"
        self.pencere.update_idletasks()
        if len(turkce_parts)-1>self.index2:
            self.index2+=1
            self.record_index=0
        else:
            self.index+=1
            if self.index>=len(self.lines) or self.index==1000:
                self.finish()
                return 
            else:
                self.record_index=0
                self.index2=0
                line=self.lines[self.index]
                parts=line.split("-")
                turkce=parts[2]
                turkce_parts=turkce.split(",")
        self.buton.focus_force()
    def finish(self):
        self.pencere.destroy()
    def load_train_model(self):
        data = self.tr.get_data()
        newdata = self.tr.augmentation(data)
        self.tr.keras_load()
        self.tr.start_training(data,newdata)
        self.test_console(True)
    def train_model(self):
        
        data = self.tr.get_data()
        newdata = self.tr.augmentation(data)
        self.tr.keras_model()
        #self.tr.load()
        self.tr.start_training(data,newdata)
        self.test_console(True)
    def load_model(self,console=True):
        self.tr.get_data(False)
        self.tr.model()
        self.tr.load()
        if console==True:
            self.test_console()
    def load_keras_model(self,console=True):
        self.tr.keras_load()
        if console==True:
            self.test_console(True)
    def test_console(self,keras=False):
        while True:
            print("test etmek için bir tuşa bas:")
            input()
            print("dinliyor")
            recording=listen(2.5)
            print("bitti")
            write("./anlık_test/test.wav", 44100, recording)
            ng, fs=  sf.read("./anlık_test/test.wav", dtype='float32')
            if keras==False:
                print("cevap:",self.tr.dinle(ng))
            else:
                print("cevap:",self.tr.keras_predict(ng))
    def test(self,indata,frames,time,status):
        """sayac2=0
        sayac=0
        
        while True:
            if (self.flag==True):
                
                write("./anlık_test/test_"+str(sayac2)+".wav", 44100, self.test_arr[sayac])
                self.flag=False
                self.answer = self.tr.dinle(self.test_arr)
                print("answer:",self.answer)
                self.answer_flag=True
                sayac+=1
                sayac2+=1
                if sayac>=5:
                    sayac=0"""
        if self.s<(self.voice_sec/self.nw)-1:
            self.test_arr[0][self.s*self.part_size:(self.s+1)*self.part_size]=indata
            self.s+=1
        elif self.s==(self.voice_sec/self.nw)-1:
            buffer = np.copy(self.test_arr[0])
            self.test_arr[0][:self.voice_size-self.part_size]=buffer[self.part_size:]
            self.test_arr[0][self.s*self.part_size:(self.s+1)*self.part_size]=indata
            self.answer = self.tr.keras_predict(self.test_arr)
            self.answer_flag=True
        

            #write("./anlık_test/test_"+str(self.s2)+".wav", 44100, self.test_arr[0])
        self.s2+=1
    def create_test_thread(self):
        #self.t1 = threading.Thread(target=lambda:self.test())
        #self.t2 = threading.Thread(target=lambda:self.live())
        #self.t2.start()
        #self.t1.start()
        #self.live()
        None
        
        
if __name__=="__main__":
    ls=listener()
    #ls.load_train_model()
    #ls.load_keras_model()
    ls.define_window()
    #ls.load_model()#tear,expend,salary
    #ls.train_model()#contest,spoil,confess,presence,stray,mildly,reward
    #save_test()

        
