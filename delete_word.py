def delete(index):
    index=int(index)
    dosya=open("belgeler/sozluk.txt","r",encoding="utf-8")
    lines = dosya.readlines()
    dosya_yedek=open("belgeler/sozluk_yedek.txt","w",encoding="utf-8")
    dosya_yedek.writelines(lines)
    dosya_yedek.close()
    dosya.close()
    lines.pop(index)
    for i in range(index,len(lines)):
        parts=lines[i].split("-")
        
        if parts[3][-1]=="\n":
            new_index=int(parts[3][:-1])-1
        lines[i]=parts[0]+"-"+parts[1]+"-"+parts[2]+"-"+str(new_index)+"\n"
    dosya_write=open("belgeler/sozluk.txt","w",encoding="utf-8")
    
    dosya_write.writelines(lines)
    
    dosya_write.close()
def re_index():
    dosya=open("belgeler/sozluk.txt","r",encoding="utf-8")
    lines = dosya.readlines()
    dosya_yedek=open("belgeler/sozluk_yedek.txt","w",encoding="utf-8")
    dosya_yedek.writelines(lines)
    dosya_yedek.close()
    dosya.close()
    for i in range(0,len(lines)):
        parts=lines[i].split("-")
        lines[i]=parts[0]+"-"+parts[1]+"-"+parts[2]+"-"+str(i)+"\n"
    dosya_write=open("belgeler/sozluk.txt","w",encoding="utf-8")
    
    dosya_write.writelines(lines)
    
    dosya_write.close()
if __name__=="__main__":
    while True:
        print("silinecek index değerini girin:")
        index=input()
        delete(index)
        #re_index()
