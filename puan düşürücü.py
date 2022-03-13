import random
dosya=open("belgeler/sozluk.txt","r+")
lines=dosya.readlines()
sayac=0
for i in lines:
    point=int(lines[sayac].split("-")[0])
    #point-=100# işlem yapmak için bunu kullan
    
    new_line=str(point)+"-"
    for q in lines[sayac].split("-")[1:]:
        new_line+=q+"-"
    new_line=new_line[:-1]
    lines[sayac]=new_line
        
    sayac+=1


dosya.close()
dosya=open("sozluk.txt","w")
dosya.writelines(lines)
dosya.close()
