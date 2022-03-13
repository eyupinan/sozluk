dosya = open("words.txt","r")
dosya_lines = dosya.readlines()
new_list=[]
for i in dosya_lines:
    #print(bytes(i,encoding="utf-8"))
    
    if i!="\n":
        sp = i.split(":")
        sp=sp[0].split(" ")
        new_list.append(sp[0].lower())
print(new_list)
print(len(new_list))
sayac=0
for i in new_list:
    for q in i:
        sayac+=1
for i in range(len(new_list)):
    new_list[i]=new_list[i]+"\n"
print("sayac:",sayac)
new_file=open("eklenecekler.txt","w")
new_file.writelines(new_list)
new_file.close()
