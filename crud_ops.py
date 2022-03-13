import find_word
import delete_word
import update_word
import create_word
import sys
def _input(st=None):
    if st!=None:
        print(st)
    
    for line in sys.stdin:
        veri = line
        break
    return veri[:-1]
if __name__=="__main__":
    while True:
        print("find:0,create:1,update:2,delete3,findByIndex:4")
        op= _input()

        if op=="0":
            word=_input("aranacak kelimeyi girin: ")
            print(find_word.find(word,True))
        elif op=="1":
            eng = _input("ingilizce değeri girin: ")
            tr = _input ("türkçe değeri girin: ")
            create_word.create(eng,tr)
        elif op=="2":
            index = _input("update edilecek index'i girin: ")
            eng = _input("ingilizce değeri girin: ")
            tr = _input ("türkçe değeri girin: ")
            point = _input ("yeni değeri girin: ")
            update_word.update(index,eng,tr,point)
        elif op=="3":
            index=_input("silinecek index'i girin: ")
            delete_word.delete(index)
        elif op=="4":
            index=_input("aranacak index'i girin: ")
            print(find_word.find_by_index(index))
        else:
            
            print(find_word.find(op,True))
