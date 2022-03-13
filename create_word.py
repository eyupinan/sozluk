import random
def create(eng,tr):
    if eng=="" or eng==None or tr=="" or tr==None:
        return
    dos=open("belgeler/sozluk.txt","r",encoding="utf-8")
    lines=dos.readlines()
    dos.close()
    dos_yedek=open("belgeler/sozluk_yedek.txt","w",encoding="utf-8")
    dos_yedek.writelines(lines)
    dos_yedek.close()
    rand= random.randint(6500,7000)
    
    lines.append(str(rand)+"-"+eng+"-"+tr+"-"+str(len(lines))+"\n")
    dos=open("belgeler/sozluk.txt","w",encoding="utf-8")
    dos.writelines(lines)
    dos.close()
