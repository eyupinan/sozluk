#!/usr/bin/env python
# -*- coding: utf-8 -*-
from tkinter import *
from random import *
from threading import Thread
import sys
import time
import std_mean as sm
import translate_api as tr
import find_word
import re
import os
import record_voice as rv
def decoder(string):
    new_latin=""
    for i in string:
        try:
            new_latin=new_latin+bytes(i,encoding='latin1').decode("cp1254")
        except:
            new_latin=new_latin+i
    
    return new_latin
class kelimeCalisma(object):
    def __init__(self):
        self.r = rv.listener()
        self.r.load_keras_model(console=False)
        self.r.create_test_thread()
        self.after_count=0
        self.kademe_sayisi=20
        self.kademe_boyut=20
        self.kademe_katsayi=1.2
        self.kademe_yontem="çarpma"
        self.kelimeler2=[]
        self.pecentage_list=self.pecentage_list_generator()
        print(self.pecentage_list)
        self.x_size=630
        self.y_size=300
        self.ekleme()
        metod=""
        kelime100="false"
        u="false"
        pencere1=Tk()
        
        pencere1.geometry("500x150+"+str(self.x_size)+"+"+str(self.y_size))
        etiket=Label(pencere1,text="sözlüğün tamamını mı yoksa son 100 kelimeyi mi görmek istersin?  ")
        etiket.pack()
        dugme5=Button(pencere1,text="tamamı",command=lambda:self.fonk3(pencere1,2))
        dugme5.pack(side=LEFT)
        dugme7=Button(pencere1,text="100 kelime",command=lambda:self.fonk3(pencere1,3))
        dugme7.pack(side=RIGHT)
        dugme6=Button(pencere1,text="bitir",command=pencere1.destroy)
        dugme6.pack() 
        pencere1.mainloop()           
    def ekleme(self):
        ekle=open("eklenecekler.txt","r")
        dosyaveri=ekle.readlines()
        if len(dosyaveri)!=0:
            yedek_ekle=open("belgeler/eklenecekler_yedek.txt","w")
            yedek_ekle.writelines(dosyaveri)
            yedek_ekle.close()
        dos=open("belgeler/sozluk.txt","r",encoding="utf-8")
        sozluk=dos.readlines()
        sozluk_yedek=open("belgeler/sozluk_yedek.txt","w",encoding="utf-8")
        sozluk_yedek.writelines(sozluk)
        sozluk_yedek.close()
        dos.close()
        count=0
        for i in sozluk:
            if len(i.split("-"))>1:
                count+=1
        dos=open("belgeler/sozluk.txt","a",encoding="utf-8")
        untranslated_words=[]
        for i in dosyaveri:

            parts= i.split("-")
            if len(parts)<2:
                
                if i[::-1][0]=="\n":
                    if find_word.find_bool(i[:-1])==False:
                        untranslated_words.append(i[:-1])
                else:
                    if find_word.find_bool(i)==False:
                        untranslated_words.append(i)
                continue
            f=randint(6500,7000)
            if (i[::-1][0]=="\n"):
                #yazılacak=i[:-1]+"-"+str(f)+"\n"
                yazılacak=str(f)+"-"+i[:-1]+"-"+str(count)+"\n"
            else:
                yazılacak=str(f)+"-"+i+"-"+str(count)+"\n"
            count+=1
            print("yaz:"+yazılacak)
            dos.write(yazılacak)
        translated=[]
        if len(untranslated_words)!=0:
            translated = tr.translate_text(untranslated_words)
        
        index=0
        for i in translated:
            translated[index]=translated[index].lower()
            f=randint(6500,7000)
            splited= i.split(",")
            if len(splited)>1:
                space_deleted=""
                for q in splited:
                    if space_deleted!="" and space_deleted[-1]!=",":
                        space_deleted+=","
                    if q[0]==" ":
                        space_deleted+=q[1:]
                    else:
                        space_deleted+=q
                translated[index]=space_deleted 
            yazılacak=str(f)+"-"+untranslated_words[index].encode().decode("utf-8") +"-"+translated[index].encode().decode("utf-8") +"-"+str(count)+"\n"
            count+=1
            index+=1
            dos.write(yazılacak)
        dos.close()
        dosya=open("eklenecekler.txt","w")
        dosya.close()
    def sayaç_updater(self,kelime100):
            f=open("belgeler/sayaç.txt","r+")
            sayac=f.read()
            sayac=int(sayac)+1
            f.close()
            f=open("belgeler/sayaç.txt","w")
            f.write(str(sayac))
            f.close()
            return sayac-1
    def relu(self,sayı):
        if sayı>0:
            return sayı
        return 0
    def yedekle(self):
        dosya=open("belgeler/sozluk.txt","r")
        dosyaliste=dosya.readlines()
        dosya.close()
        try:
            print("yedekleme yapılması gerekiyor.")
            sayac=0
            while True:
                if not os.path.isfile("belgeler/backup/sozluk_yedek"+str(sayac)+".txt"):
                    
                    dosya_yedek=open("belgeler/backup/sozluk_yedek"+str(sayac)+".txt","w")
                    dosya_yedek.writelines(dosyaliste)
                    dosya_yedek.close()
                    print("yedekleme yapıldı.")
                    break
                sayac+=1
        except Exception as e:
            print(e)
            exit()
        
    def rakam(self,a,rastgele,x,onay,kelime100,doğrusayısı,yanlışsayısı):
        dosya=open("belgeler/sozluk.txt","r")
        dosyaliste=dosya.readlines()
        count=0
        for i in dosyaliste:
            count+=1
        dosya.close()
        if kelime100!="true":
            if onay=="true":
                rd=randint(950,1050)
                print("doğru :",self.doğrusayısı,"yanlış :",self.yanlışsayısı)
                self.label_info8["text"]="değişim : "+ str(-rd-int(+self.relu( (self.mean-int(a[0]))/4)))
                a[0]=int(a[0])-int(rd+self.relu((self.mean-int(a[0]))/4))
                self.label_info9["text"]="yeni değer : "+ str(a[0])
            elif onay=="false":
                print("doğru :",self.doğrusayısı,"yanlış :",self.yanlışsayısı)
                if int(a[0])<4500:
                    self.label_info8["text"]="değişim : "+ str(int(600+self.relu( (self.mean-int(a[0]))/3)))
                    a[0]=int(a[0])+int(750+self.relu( (self.mean-int(a[0]))/3))
                elif int(a[0])<5000:
                    self.label_info8["text"]="değişim : "+ str(int(400+self.relu( (self.mean-int(a[0]))/3)))
                    a[0]=int(a[0])+int(500+self.relu( (self.mean-int(a[0]))/3))
                else:
                    self.label_info8["text"]="değişim : "+ str(int(300+self.relu( (self.mean-int(a[0]))/3)))
                    a[0]=int(a[0])+int(400+self.relu( (self.mean-int(a[0]))/3))
                self.label_info9["text"]="yeni değer : "+ str(a[0])
        if kelime100=="true":
            if onay=="true":
                print("doğru :",self.doğrusayısı,"yanlış :",self.yanlışsayısı)
                a[0]=int(a[0])-40
            elif onay=="false":
                print("doğru :",self.doğrusayısı,"yanlış :",self.yanlışsayısı)
                a[0]=int(a[0])+30
        if a[0]<1000:
            a[0]=1000
        if a[0]>9999:
            a[0]=9999
        veriler=dosyaliste[rastgele].split("-")
        dosyaliste[rastgele]=str(a[0])+"-"+veriler[1]+"-"+veriler[2]+"-"+veriler[3]
        #dosyaliste[rastgele]=str(a[0])+"-"+str(rastgele)+"\n"
        
        self.sayac=self.sayaç_updater(kelime100)
        self.label_info1["text"]="sayaç : "+str(self.sayac)
        kalan=self.sayac%20
        if kalan==0:
            print("yedekle geldi")
            self.yedekle()
            std,mean = sm.calculate_std()
            print("std and mean: ",std,mean)
            self.mean=mean
            self.label_info5["text"]="mean : "+str(mean)
            fark = int(5000 - mean)
            
            k=0
            asdf=0
            for i in dosyaliste:
                satır=dosyaliste[k].split("-")
                değer=satır[0]
                değer=int(değer)
                #sıra=satır[1][0:-1]
                try:
                    yeni_deger = değer
                    if mean<2500 or mean>7500:
                        yeni_deger = değer+fark
                    if yeni_deger<1000:
                        yeni_deger=1000
                    if x!=0:
                        if asdf<10:
                            asdf+=1
                            print("fark:",fark," değer:",değer)
                            print("x:",x," yeni değer:",yeni_deger)
                            print("eklenen değer:",int(pow(9999-yeni_deger,2)/pow(x,2)))
                        yeni_deger+=int(pow(9999-yeni_deger,2)/pow(x,2))
                    if yeni_deger>9999:
                        yeni_deger=9999
                    dosyaliste[k]=str(yeni_deger)+"-"+satır[1]+"-"+satır[2]+"-"+satır[3]
                except Exception as e:
                    print("Hata olustu:")
                    print(e)
                    print("hata index:",k)
                    print("dosya index degeri:",dosyaliste[k])
                k+=1
            
        count2=0
        
        for i in dosyaliste:
            count2+=1
        if count2!=count:
            print("kelime kayıp!")
        else:
            dosya=open("belgeler/sozluk.txt","w")
            dosya.writelines(dosyaliste)
            dosya.close()
            if kalan==0:
                std,mean = sm.calculate_std()
                self.mean=mean
                self.label_info5["text"]="mean : "+str(mean)
    def percent(self,percentage_list):
        rand =uniform(0, 1)
        index=0
        total_percentage=0
        for i in percentage_list:
            if index==0:
                if rand<=(percentage_list[index]/100):
                    return index+1
            else:
                if rand<=((total_percentage+percentage_list[index])/100) and rand>(total_percentage/100):
                    return index+1
            total_percentage+=percentage_list[index]
            index+=1
    def pecentage_list_generator(self):
        bölen=0
        anlık=1
        katsayi_list=[]
        for i in range(self.kademe_sayisi):
           bölen+=anlık
           katsayi_list.append(anlık)#her adım için katsayıyı kaydediyor
           if self.kademe_yontem=="çarpma":
               anlık*=self.kademe_katsayi
           elif self.kademe_yontem=="toplama":
               anlık+=self.kademe_katsayi
        percentage_list=[]
        for i in range(self.kademe_sayisi,0,-1):
            percentage_list.append(100/bölen*katsayi_list[i-1])
        total=0
        for i in percentage_list:
            total+=i
        print("total:",total)
        while True:
            if total<100:
                percentage_list[0]+=100-total
                total=0
                for i in percentage_list:
                    total+=i
            else:
                break
        return percentage_list
    def rand(self,y=0):
        kademe=randint(0,13)
        dosya=open("belgeler/sozluk.txt","r",encoding="utf8")
        dosyaliste=dosya.readlines()
        dosya.close()
        x=0
        for i in dosyaliste:
            x+=1   
        self.kademe_boyut=20
        self.kademe_boyut=int(self.kademe_boyut)*(-1)
        listesıralı=sorted(dosyaliste[y:])
        rastgele=randint(0,-self.kademe_boyut-1)
        
        kademe=self.percent(self.pecentage_list)
        if kademe!=1:
            rastgeleliste=listesıralı[self.kademe_boyut*kademe:self.kademe_boyut*(kademe-1)]
        else:
            rastgeleliste=listesıralı[self.kademe_boyut*kademe:]
        self.label_info6["text"]="kademe " + str(kademe)
        self.label_info7["text"]="rastgele : " + str(rastgele)
         
        try:
            a=rastgeleliste[rastgele].split("-")
            print("deger:",a[0])
            self.label_info9["text"]="değer : " + str(a[0])
            #self.label_info8["text"]="değişim : " 
        except:
            rastgele=randint(0,-self.kademe_boyut-1)
            print("aranan değer bulunamadı!!! yeni rastgele:",rastgele)
            rastgeleliste=listesıralı[self.kademe_boyut+1:]
            print("kademe 1")
            print("rastgele",rastgele)
            a=rastgeleliste[rastgele].split("-")
            print("deger:",a[0])
            self.label_info9["text"]="değer : " + str(a[0])
            self.label_info8["text"]="değişim : " 
        try:
            sıra=a[3][:-1]
        except:
            print("eksik kelime geldi!")
            print("hatali ifade:",a)
            return 0
        sıra=int(sıra)
        print("sıra:",sıra)
        #return randint(0,131)
        return sıra
    def fonk1(self,pencere,giris):
        metod=giris.get()
        dosya=open("belgeler/arabellek.txt","w")
        dosya.write(metod)
        dosya.close()
        self.x_size=pencere.winfo_x()
        self.y_size=pencere.winfo_y()
        pencere.destroy()
    def fonk4(self):
        self.wrong_ans()
        self.x_size=self.pencere.winfo_x()
        self.y_size=self.pencere.winfo_y()

    def fonk5(self):
                #self.label_info2["text"]="toplam : "+str(self.after_count)
                self.after_count+=1
                girilen=self.giris.get()
                self.giris.delete(0, END)
                string=decoder(girilen)
                self.giris.insert(END, string)
                if girilen+"\n" in self.kelimeler2 or girilen in self.kelimeler2:
                    self.after_count=0
                    self.correct_ans()                    
                    self.x_size=self.pencere.winfo_x()
                    self.y_size=self.pencere.winfo_y()
                if len(girilen)>4 and self.metod=="1":
                    
                    for kelime in self.kelimeler2:
                        if girilen==kelime[:len(girilen)]:
                            self.after_count=0
                            self.correct_ans()                    
                            self.x_size=self.pencere.winfo_x()
                            self.y_size=self.pencere.winfo_y()
                if self.after_count<50:
                    verify=False
                    if girilen=="":
                        verify=True
                    for kelime in self.kelimeler2:
                        verify2=True
                        for i in range(len(girilen)):
                            if len(kelime)>i:
                                if girilen[i]!=kelime[i]:
                                    verify2=False
                        if verify2==True:
                            verify=True
                    if verify==False:
                        self.giris.delete(0, 'end')
                    
                self.kontrol_buton.after(10,self.fonk5)
    def correct_ans(self):
        self.show_word.pack_forget()
        self.giris.pack_forget()
        self.dugme_kontrol.pack_forget()
        self.doğrusayısı+=1
        self.total=self.doğrusayısı+self.yanlışsayısı
        self.rakam(self.a,self.rastgele,self.x,"true","false",self.doğrusayısı,self.yanlışsayısı)
        self.label_info2["text"]="toplam : "+str(self.total)
        self.label_info3["text"]="doğru : "+str(self.doğrusayısı) 
        self.correct_answer.pack()
        self.correct_answer_word["text"]="doğrusu {}".format(self.answer)
        self.correct_answer_word.pack()
        self.new_button.focus_force()
        self.new_button.pack()
        self.ask()
    def wrong_ans(self):
        self.show_word.pack_forget()
        self.giris.pack_forget()
        self.dugme_kontrol.pack_forget()
        self.yanlışsayısı+=1
        self.total=self.doğrusayısı+self.yanlışsayısı
        self.rakam(self.a,self.rastgele,self.x,"false","false",self.doğrusayısı,self.yanlışsayısı)
        self.label_info2["text"]="toplam : "+str(self.total)
        self.label_info4["text"]="yanlış : "+str(self.yanlışsayısı)
        self.wrong_answer.pack()
        self.wrong_answer_word["text"]="doğrusu {}".format(self.answer)
        self.wrong_answer_word.pack()
        self.new_button.focus_force()
        self.new_button.pack()
        
    def ask(self):
        int_var = IntVar()
        dosya=open("belgeler/sozluk.txt","r",encoding="utf8")
        sozluk=dosya.readlines()
        dosya.close()
        self.giris.delete(0, 'end')
        self.correct_answer.pack_forget()
        self.correct_answer_word.pack_forget()
        self.wrong_answer.pack_forget()
        self.wrong_answer_word.pack_forget()
        self.new_button.pack_forget()
        self.rastgele=self.rand()
        self.a=[int(sozluk[self.rastgele].split("-")[0])]
        self.ingilizce=sozluk[self.rastgele].split("-")[1].lower()
        self.turkce=sozluk[self.rastgele].split("-")[2].lower()      
        
        
        if self.metod=="1":
            self.asked=self.ingilizce
            self.answer=self.turkce
        elif self.metod=="2":
            self.asked=self.turkce
            self.answer=self.ingilizce
        kelimeler=self.asked.split(",")
        self.kelimeler2=self.answer.split(",")
        ö=0
        for i in kelimeler:
            ö+=1    
        say=randint(0,ö-1)
        #etiket = Label(self.pencere,text = "kelime {}".format(kelimeler[say]))
        self.show_word["text"]="kelime {}".format(kelimeler[say])
        self.show_word.pack()
        self.giris.focus_force()
        self.giris.pack()
        self.dugme_kontrol.pack()
    def fonk2(self,y=0,kelime100="false"):
        global giris
        #mainloop()
        dosya=open("belgeler/sozluk.txt","r")
        sozluk=dosya.readlines()
        dosya.close()
        pencere=Tk()
        pencere.wm_attributes("-topmost",1)
        
        pencere.geometry("400x150+"+str(self.x_size)+"+"+str(self.y_size))
        etiket=Label(pencere,text="ingilizceden türkçeye için 1 türkçeden ingilizceye için 2  yaz:  ")
        etiket.pack()
        giris=Entry(pencere)
        giris.pack()
        
        dugme3=Button(pencere,text="tamam",command= lambda:self.fonk1(pencere,giris))
        dugme3.pack()
        #dugme4=Button(text="bitir",command=pencere.destroy)
        #dugme4.pack()
        pencere.mainloop()
        dosya=open("belgeler/arabellek.txt","r")
        self.metod=dosya.read()
        dosya.close()
        self.x=0
        for i in sozluk:
            self.x+=1
        self.doğrusayısı=0
        self.yanlışsayısı=0
        self.pencere = Tk()
        self.pencere.wm_attributes("-topmost",1)
        self.pencere.geometry("300x150+"+str(self.x_size)+"+"+str(self.y_size))
        self.show_word = Label(self.pencere,text ="")
        self.giris = Entry(self.pencere)
        self.giris.focus_force()
        
        self.kontrol_buton=Button(self.pencere)
        self.dugme_kontrol = Button(self.pencere,text="kontrol", command =lambda:self.fonk4())
        self.correct_answer=Label(self.pencere,text="girilen kelime doğru!! ",fg="green")
        self.correct_answer_word=Label(self.pencere,text="")
        self.wrong_answer=Label(self.pencere,text="girilen kelime yanlış!!",fg="red")
        self.wrong_answer_word=Label(self.pencere,text="")
        int_var = IntVar()
        self.new_button=Button(self.pencere,text="bitir",command=lambda: self.ask())
        self.kontrol_buton.after(100,self.fonk5)
        
        self.ask()
        print("geldi")
        self.pencere.mainloop()
    def kontrol(self):
        while True:
            try:
                #girilen=self.giris.get()

                for i in self.kelimeler2:
                    print(i)
                self.doğru="false"
                if girilen+"\n" in self.kelimeler2 or girilen in self.kelimeler2:
                    self.doğru="true"

            except Exception as e:
                print(e)
    def fonkinfo(self):
        self.pencere33=Tk()
        self.pencere33.wm_attributes("-topmost",1)
        self.pencere33.geometry("200x220+"+str(630)+"+"+str(50))
        self.label_info1=Label(self.pencere33,text="")
        self.label_info1.pack()
        self.label_info2=Label(self.pencere33,text="")
        self.label_info2.pack()
        self.label_info3=Label(self.pencere33,text="")
        self.label_info3.pack()
        self.label_info4=Label(self.pencere33,text="")
        self.label_info4.pack()
        self.label_info5=Label(self.pencere33,text="")
        self.label_info5.pack()
        self.label_info6=Label(self.pencere33,text="")
        self.label_info6.pack()
        self.label_info7=Label(self.pencere33,text="")
        self.label_info7.pack()
        self.label_info8=Label(self.pencere33,text="")
        self.label_info8.pack()
        self.label_info9=Label(self.pencere33,text="")
        self.label_info9.pack()
        self.button_ayar=Button(self.pencere33,text="ayarlar",command=lambda:self.ayar())
        self.button_ayar.pack()
        std,mean = sm.calculate_std()
        self.mean=mean
        print("std and mean: ",std,mean)
        self.label_info5["text"]="mean : "+str(mean)
        mainloop()
    def ayar(self):
        
        pencere=Tk()
        pencere.geometry("200x200+"+str(630)+"+"+str(500))
        lb1=Label(pencere,text="kademe sayısı:")
        lb1.pack()
        ent1=Entry(pencere)
        ent1.pack()
        lb2=Label(pencere,text="kademe boyut:")
        lb2.pack()
        ent2=Entry(pencere)
        ent2.pack()
        lb3=Label(pencere,text="kademe katsayı:")
        lb3.pack()
        ent3=Entry(pencere)
        ent3.pack()
        def save():
            kademe_sayisi=ent1.get()
            kademe_boyut=ent2.get()
            kademe_katsayi=ent3.get()
            if kademe_boyut!="":
                self.kademe_boyut=int(kademe_boyut)
                print("yeni boyut:",self.kademe_boyut)
            if kademe_sayisi!="":
                self.kademe_sayisi=int(kademe_sayisi)
            if kademe_katsayi!="":
                self.kademe_katsayi=float(kademe_katsayi)
            self.pecentage_list=self.pecentage_list_generator()
            print(self.pecentage_list)
            pencere.destroy()
            
        but=Button(pencere,text="save",command=lambda:save())
        but.pack()
        pencere.mainloop()
    def ai_listener(self):
        while True:
            if self.r.answer_flag==True:
                print(self.r.answer)
                key_list=self.r.answer.keys()
                for i in key_list:
                    if i+"\n" in self.kelimeler2 or i in self.kelimeler2:
                        self.after_count=0
                        self.correct_ans()                    
                        self.x_size=self.pencere.winfo_x()
                        self.y_size=self.pencere.winfo_y()
                        break
            time.sleep(0.15)
    def fonk3(self,pencere,y):
        pencere.destroy()
        thread_info=Thread(target=self.fonkinfo,args=())
        thread_info.start()
        if y==2:
            thread1=Thread(target=self.fonk2,args=())
            thread2=Thread(target=lambda:self.ai_listener(),args=())
            thread1.start()
            thread2.start()
            thread1.join()
            thread_info.join()
            thread2.join()
        else:
            global x
            kelime100="true"
            türk=open("belgeler/sozluk.txt","r")
            turkce=türk.readlines()
            x=0
            for i in turkce:
                x+=1
            
            y=x-100
            türk.close()
            print("geldi2")
            thread1=Thread(target=lambda:self.fonk2(y,kelime100),args=())
            thread2=Thread(target=lambda:self.ai_listener(),args=())
            thread1.start()
            thread2.start()
            thread1.join()
            thread_info.join()
            thread2.join()
if __name__=="__main__":
    ornek=kelimeCalisma()
     


        
        
        
        
        
    
