import re
def find(string_list,re_usage=False):
    if not isinstance(string_list, list):
        string_list=[string_list]
    dosya=open("belgeler/sozluk.txt","r",encoding="utf-8")
    lines=dosya.readlines()
    dosya.close()
    sayac=0
    liste=[]
    for i in lines:
        parts=i.split("-")
        eng_parts=parts[1].split(",")
        tr_parts=parts[2].split(",")
        for string in string_list:
            if re_usage==False:
                for q in eng_parts:
                    if string==q:
                        liste.append(i[:-1])
                for q in tr_parts:
                    if string==q:
                        liste.append(i[:-1])
            else:
                for q in eng_parts:
                    if re.search(string,q):
                        liste.append(i[:-1])
                for q in tr_parts:
                    if re.search(string,q):
                        liste.append(i[:-1])
                    
    return liste
def find_by_index(index):
    index=int(index)
    dosya=open("belgeler/sozluk.txt","r",encoding="utf-8")
    lines=dosya.readlines()
    dosya.close()
    return lines[index]
def find_bool(string_list):
    if len(find(string_list))==0:
        return False
    return True
if __name__=="__main__":
    while True:
        print("enter a word")
        word=input()
        print(find(word,True))
    
