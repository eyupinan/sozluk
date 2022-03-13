def sort():
    dosya=open("belgeler/sozluk.txt","r",encoding="utf-8")
    lines=dosya.readlines()
    listesıralı=sorted(lines)[::-1]
    sayac=0
    for i in listesıralı:
        print(str(sayac)+"-"+i)
        sayac+=1
sort()
input()
