import find_word
def update(index,eng=None,tr=None,point=None):
    word = find_word.find_by_index(index)
    parts= word.split("-")
    if eng!=None and eng!="":
        parts[1]=eng
    if tr!=None and tr!="":
        parts[2]=tr
    if point!=None and point!="":
        try:
            point=int(point)
            parts[0]=str(point)
        except:
            None
    dosya=open("belgeler/sozluk.txt","r",encoding="utf-8")
    lines=dosya.readlines()
    dosya.close()
    lines[int(index)]=parts[0]+"-"+parts[1]+"-"+parts[2]+"-"+parts[3]
    dosya=open("belgeler/sozluk.txt","w",encoding="utf-8")
    dosya.writelines(lines)
    dosya.close()
