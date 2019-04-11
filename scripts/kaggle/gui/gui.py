import tkinter as tk
import time
import numpy as np
from PIL import Image,ImageTk

class Data:
    def __init__(self):
        self.classifier = np.random.randint(0,3)


class GUI:
    def __init__(self,parent):
        
        self.Top = parent
        self.Top.config(bg = "black")
        Top.geometry("1200x700")
        
        ##  BrainWave display
        
        self.frameWave = tk.Frame(self.Top,height = 700,width = 600,bg = 'yellow',bd = 0)
        self.frameWave.pack(side="left",fill="both",expand=1)       
        self.Refresh = tk.Button(self.frameWave, text = "Refresh", command = self.resetCanvas);
        self.Refresh.pack(fill = 'both')   
            
            ## Display Canvas (After the refresh button)
            
        self.frameEEG = tk.Frame(self.frameWave,height = 650,width = 600,bg = 'yellow',bd = 0)
        self.frameEEG.pack(fill = 'both',expand=1,side='bottom')
        
            ## Header Canvas 
        
        self.Header = tk.Canvas(self.frameEEG,height = 50,width = 600,bd=0)
        self.Header.create_text(300,30,text='Given Data ')
        self.Header.update_idletasks()
        self.Header.pack(fill='x')
        
            ## Plot Canvas
        
        self.C = tk.Canvas(self.frameEEG,bg = "white", height = 600, width = 600,bd=0)
        self.C.pack(fill = 'x',expand=1,side='bottom')
        self.points = 150
        self.height= 20
        self.data = Data()

        ## Right frame 
        
        self.frameAction = tk.Frame(self.Top,height = 700,width = 600,bg = 'red',bd = 0)
        self.frameAction.pack(side="right",fill='both',expand=1)
            
        ## Top canvas in right frame
        
        # self.c = tk.Canvas(self.frameEEG,height = 30,width = 600,bd=0)
        # self.c.create_text(300,30,text='Classifier Output ')
        # self.c.update_idletasks()
        # self.c.pack(fill='x')

        self.Ctop = tk.Canvas(self.frameAction,bg = "black", height = 350, width = 600,bd=0)
        self.Ctop.pack(fill = 'x',side="top")
        
        ##  Bottom canvas in right frame
   
        self.Cbottom = tk.Canvas(self.frameAction,bg = "black", height = 350, width = 600,bd=0)
        self.Cbottom.pack(side = "bottom",fill = "x")
        #self.picbottom()            

        #Resize image
        
        basewidth = 600

        #Open Images
        
        self.img_t1 = Image.open("t1.png")
        wpercent = (basewidth/float(self.img_t1.size[0]))
        hsize = int((float(self.img_t1.size[1])*float(wpercent)))
        self.img_t1 = self.img_t1.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_t1 = ImageTk.PhotoImage(self.img_t1)
        
        self.img_t2 = Image.open("t2.png")
        wpercent = (basewidth/float(self.img_t2.size[0]))
        hsize = int((float(self.img_t2.size[1])*float(wpercent)))
        self.img_t2 = self.img_t2.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_t2 = ImageTk.PhotoImage(self.img_t2)
        
        self.img_t3 = Image.open("t3.png")
        wpercent = (basewidth/float(self.img_t3.size[0]))
        hsize = int((float(self.img_t3.size[1])*float(wpercent)))
        self.img_t3 = self.img_t3.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_t3 = ImageTk.PhotoImage(self.img_t3)
                
        self.img_b1 = Image.open("b1.png")
        wpercent = (basewidth/float(self.img_b1.size[0]))
        hsize = int((float(self.img_b1.size[1])*float(wpercent)))
        self.img_b1 = self.img_b1.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_b1 = ImageTk.PhotoImage(self.img_b1)
        
        self.img_b2 = Image.open("b2.png")
        wpercent = (basewidth/float(self.img_b2.size[0]))
        hsize = int((float(self.img_b2.size[1])*float(wpercent)))
        self.img_b2 = self.img_b2.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_b2 = ImageTk.PhotoImage(self.img_b2)
        
        self.img_b3 = Image.open("b3.png")
        wpercent = (basewidth/float(self.img_b3.size[0]))
        hsize = int((float(self.img_b3.size[1])*float(wpercent)))
        self.img_b3 = self.img_b3.resize((basewidth,hsize), Image.ANTIALIAS)
        self.img_b3 = ImageTk.PhotoImage(self.img_b3)
        
    def resetCanvas(self):
        self.C.configure(bg = "grey")
        self.frameWave.update_idletasks()
        time.sleep(0.1)
        self.C.configure(bg = "white")
        self.frameWave.update_idletasks()
        self.refresh()
    
    def refresh(self):
        try:
             while True:
                self.data = Data()
                classes = ["StageID = "," Raised Hands = ","Visited Resources = ","Announcements View = ", "Discussion = ", "Parent Answering Survey = ", "Parent School Satisfaction = ", "Student Absence Days = ", "Class = "]
                number = np.arange(50,651,550/8)
                text = [chr(65+i) for i in range(8)]
                points = np.arange(0,901,900/self.points)
                self.C.delete("all")
                self.Ctop.delete("all")
                self.Cbottom.delete("all")
                self.Ctop.create_text(130,20,fill="white",font="Courier")#, text="Classifier Output")
                self.Ctop.update_idletasks()
                self.Cbottom.create_text(100,20,fill="white",font="Courier", text="")
                self.Cbottom.update_idletasks()
    
                self.C.create_text(230,number[0],anchor = 'w',text = classes[0]+text[0])
                self.C.create_text(200,number[1],anchor = 'w',text = classes[1]+text[1])
                self.C.create_text(190,number[2],anchor = 'w',text = classes[2]+text[2])
                self.C.create_text(180,number[3],anchor = 'w',text = classes[3]+text[3])
                self.C.create_text(230,number[4],anchor = 'w',text = classes[4]+text[4])
                self.C.create_text(150,number[5],anchor = 'w',text = classes[5]+text[5])
                self.C.create_text(150,number[6],anchor = 'w',text = classes[6]+text[6])
                self.C.create_text(150,number[7],anchor = 'w',text = classes[7]+text[7])
                self.C.update_idletasks()

                a = self.data.classifier
                if a == 0 :
                    self.top_img(self.img_t1)
                    self.bottom_img(self.img_b1)
                elif a == 1:
                    self.top_img(self.img_t2)
                    self.bottom_img(self.img_b2)
                elif a == 2:
                    self.top_img(self.img_t3)
                    self.bottom_img(self.img_b3)
                self.C.update_idletasks()
                self.Top.after(500)    
        except Exception as e:
            print(e)
    def top_img(self,Img1):
        self.Ctop.create_image(300,175,image=Img1,anchor="center");
        self.Ctop.update_idletasks()
    def bottom_img(self,Img2):
        self.Cbottom.create_image(300,175,image=Img2,anchor="center");
        self.Cbottom.update_idletasks()

    
                      
Top = tk.Tk()
window = GUI(Top)
Top.mainloop()