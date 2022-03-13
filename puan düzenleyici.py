import random
def regulator():
    dosya = open("belgeler/sozluk.txt","r",encoding="utf-8")
    lines = dosya.readlines()
    dosya.close()
    new_lines=[]
    for i in lines:
        parts=i.split("-")
        new_lines.append(str(random.randint(6500,7500))+"-"+parts[1]+"-"+parts[2]+"-"+parts[3])
    
    dosya = open("belgeler/sozluk.txt","w",encoding="utf-8")
    dosya.writelines(new_lines)
    dosya.close()
regulator()
