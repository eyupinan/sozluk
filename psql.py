import psycopg2
import time
        
class connection:
    def __init__(self):
        self.conn = psycopg2.connect(database="dictionary", user="postgres", password="Eyup_inan5", host="127.0.0.1", port="5432")
    def rollback(self):
        cur=self.conn.cursor()
        cur.execute("ROLLBACK")
        self.conn.commit()
    def select_relation(self,english_name=None,turkish_name=None,limit=None):
        cur = self.conn.cursor()
        st = "SELECT turkish_name,english_name,point  from relation"
        query_list=[]
        if english_name!=None:
            query_list.append("english_name = \'"+english_name+"\'")
        if turkish_name!=None:
            query_list.append("turkish_name = \'"+turkish_name+"\'")
        if (len(query_list)>0):
            st=st+" where "
            st+=query_list[0]
            for i in range(1,len(query_list)):
                st+=" and "
                st+=query_list[i]
        st+= " order by point desc "
        if limit!=None:
            st+=" limit "+ str(limit)
        cur.execute(st)
        rows = cur.fetchall()
        cur.close()
        if len(rows)==1:
            return rows[0]
        else:
            return rows
    def select_english_word(self,english_name=None):
        cur = self.conn.cursor()
        st = "SELECT name,description from english"
        query_list=[]
        if english_name!=None:
            query_list.append("name = \'"+english_name+"\'")
        if (len(query_list)>0):
            st=st+" where "
            st+=query_list[0]
            for i in range(1,len(query_list)):
                st+=" and "
                st+=query_list[i]
        cur.execute(st)
        rows = cur.fetchall()
        if len(rows)==1:
            return rows[0]
        else:
            return rows
    def select_turkish_word(self,turkish_name=None):
        cur = self.conn.cursor()
        st = "SELECT name,description from turkish"
        query_list=[]
        if turkish_name!=None:
            query_list.append("name = \'"+turkish_name+"\'")
        if (len(query_list)>0):
            st=st+" where "
            st+=query_list[0]
            for i in range(1,len(query_list)):
                st+=" and "
                st+=query_list[i]
        cur.execute(st)
        rows = cur.fetchall()
        cur.close()
        if len(rows)==1:
            return rows[0]
        else:
            return rows
    def update_relation(self,english_name=None,turkish_name=None,new_point=None):
        cur = self.conn.cursor()
        if (new_point==None):
            return
        if(type(new_point)!=int):
            return
        st="update relation set point="+ str(new_point)
        query_list=[]
        if english_name!=None:
            query_list.append("english_name = \'"+english_name+"\'")
        if turkish_name!=None:
            query_list.append("turkish_name = \'"+turkish_name+"\'")
        if (len(query_list)>0):
            st=st+" where "
            st+=query_list[0]
            for i in range(1,len(query_list)):
                st+=" and "
                st+=query_list[i]
        cur.execute(st)
        self.conn.commit()
    def insert_english_word(self,name,description=None):
        cur = self.conn.cursor()
        if name==None:
            return
        
        if description!=None:
            st="insert into english(name, description) values ('"+name+"', '"+description+"')"
        else:
            st="insert into english(name) values ('"+name+"')"
        cur.execute(st)
        self.conn.commit()
        cur.close()
    def insert_turkish_word(self,name,description=None):
        cur = self.conn.cursor()
        if name==None:
            return
        
        if description!=None:
            st="insert into turkish(name, description) values ('"+name+"', '"+description+"')"
        else:
            st="insert into turkish(name) values ('"+name+"')"
        cur.execute(st)
        self.conn.commit()
    def insert_relation(self,english_name,turkish_name,point=None):
        if point == None:
            return

            
        cur = self.conn.cursor()
        st="insert into relation(english_name,turkish_name,point) values ("
        st+="'"+english_name+"','"+turkish_name+"',"+str(point)+")"
        cur.execute(st)
        self.conn.commit()
        cur.close()
        
class converter:
    def __init__(self):
        self.con = connection()
    def from_txt(self):
        dosya= open("belgeler/sozluk.txt","r",encoding="utf-8")
        lines=dosya.readlines()
        sayac=0
        for i in lines:
            parts= i.split("-")
            english_parts=parts[1].split(",")
            turkish_parts=parts[2].split(",")
            for eng in english_parts:
                try:
                    if (len(self.con.select_english_word(eng.replace("'","\\'")))==0):
                        self.con.insert_english_word(eng)
                except Exception as e:
                    print(e)
                    self.con.rollback()
            for tr in turkish_parts:
                try:
                    if (len(self.con.select_turkish_word(tr.replace("'","\\'")))==0):
                        self.con.insert_turkish_word(tr)
                except Exception as e:
                    print(e)
                    self.con.rollback()
            for eng in english_parts:      
                for tr in turkish_parts:
                    try:
                        self.con.insert_relation(eng,tr,int(parts[0]))
                    except Exception as e:
                        print(e)
                        self.con.rollback()
                    
            sayac+=1
            
            

    
if __name__=="__main__":
    convert = converter()
    convert.from_txt()
    




