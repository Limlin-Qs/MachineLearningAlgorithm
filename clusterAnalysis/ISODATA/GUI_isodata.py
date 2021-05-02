from tkinter import *
import ISODATA.isodata as isodata
import ISODATA.iso_graphic as iso_graphic


class GUI:
    def __init__(self, master):
        self.master = master

        # configuring window options
        # master.configure(background='#a0adb8')
        master.minsize(width=400, height=640)

        # self.origin()
        isodata.clusterize()
        iso_graphic.origin()
        iso_graphic.clusterized()

        Label(master, text='Original points').place(x=10, y=280)
        Label(master, text='Clusterized points').place(x=10, y=600)

        self.canvas1 = Canvas(master, width=360, height=260)
        self.img1 = PhotoImage(file='origin.png')
        self.disp1 = self.img1.subsample(2, 2)
        self.canvas1.create_image(0, 0, image=self.disp1, anchor="nw")
        self.canvas1.place(x=20, y=10)

        self.canvas2 = Canvas(master, width=360, height=260)
        self.img2 = PhotoImage(file='clusters.png')
        self.disp2 = self.img2.subsample(2, 2)
        self.canvas2.create_image(0, 0, image=self.disp2, anchor="nw")
        self.canvas2.place(x=20, y=330)

    def origin(self):
        isodata.clusterize()
        iso_graphic.origin()
        iso_graphic.clusterized()

    def clusterized(self):

        self.canvas2 = Canvas(width=400, height=300)
        self.img2 = PhotoImage(file='clusters.png')
        self.disp2 = self.img2.subsample(2, 2)
        self.canvas2.create_image(0, 0, image=self.disp2, anchor="nw")
        self.canvas2.place(x=250, y=330)


root = Tk()
root.title("Isodata algorithm")
dic = GUI(root)
root.mainloop()
