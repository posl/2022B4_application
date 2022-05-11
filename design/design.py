from audioop import reverse
from tkinter import *
import tkinter as tk
import tkinter.ttk as ttk

import numpy as np
import random
import math
import sys
import os
sys.path.append(os.path.abspath(".."))
import board


import pygame




class StartPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = master
        self.configure(bg="#881111")
        self.grid(row=0, column=0, sticky="nsew")
        self.label = tk.Label(self, text="Othello", font = (master.font_name, 50), fg="#119911", bg="#881111")
        self.label.pack(anchor="center", expand=True)
        #self.label.place(x=200, y=50)
        self.button1 = tk.Button(self, text="Play", font = (master.font_name, 50), command=lambda:self.goto_option_page())
        self.button1.pack(anchor="center", expand=True)
        #self.button1.place(x=200, y=200)
        self.button2 = tk.Button(self, text="Quit", font = (master.font_name, 50), command=lambda:master.quit())
        self.button2.pack(anchor="center", expand=True)
        #self.button2.place(x=200, y=270)

    def goto_option_page(self):
        self.master.option_page.tkraise()


class OptionFrame(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = master
        self.configure(bg="#992299")
        self.grid(row=0, column=0, sticky="nsew")

        self.combo_menus = ("手動", "COM-A(ランダム)")

        self.label1 = tk.Label(self, text="Player1", fg="#999999")
        self.label1.place(x=10, y=30)

        self.label2 = tk.Label(self, text="Player2", fg="#999999")
        self.label2.place(x=10, y=90)

        self.combobox1 = ttk.Combobox(self, height=3, values = self.combo_menus, state="readonly")
        self.combobox1.place(x=200, y=30 )
        self.combobox1.current(0)

        self.combobox2 = ttk.Combobox(self, height=3, values = self.combo_menus, state="readonly")
        self.combobox2.place(x=200, y=90 )
        self.combobox2.current(1)

        self.button1 = tk.Button(self, text="Next", font = (master.font_name, 50), command=lambda:self.start_game())
        self.button1.place(x=450, y=380)

    def start_game(self):
        self.master.board.reset()
        self.master.game_page.player1 = self.combobox1.current()
        self.master.game_page.player2 = self.combobox2.current()
        player1_plan = 0
        if self.combobox1.current() == 0:
            print("先攻：",0)
            player1_plan = self.master.com_random
        elif self.combobox1.current() == 1:
            print("先攻：",1)
            player1_plan = self.master.com_random
        if self.combobox2.current() == 0:
            print("後攻",0)
            player2_plan = self.master.com_random
        elif self.combobox2.current() == 1:
            print("後攻",1)
            player2_plan = self.master.com_random
        self.master.board.set_plan(player1_plan, player2_plan)
        self.master.game_page.canvas_update()
        self.master.game_page.tkraise()
        self.master.after(1000, self.master.game_page.button1_click)




class GamePage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = master
        self.configure(bg="#992299")
        self.grid(row=0, column=0, sticky="nsew")

        self.canvas_width = 400
        self.canvas_height = 400

        self.cell_width = (self.canvas_width-10) // 8
        self.cell_height = (self.canvas_height-10) // 8

        self.player1 = 0
        self.player2 = 0

        self.game_still_cont = 0
        #self.game_still_cont_check = 0


        self.game_canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height)
        self.game_canvas.place(x=100, y=50)
        self.game_canvas.bind("<Button-1>", self.cell_click)

        self.counter_bar = tk.Canvas(self, width=self.canvas_width, height=30)
        self.counter_bar.place(x=100, y=5)

        self.black_conter_label = tk.Label(self, text="B00", fg="#111111", bg="#808080", font = (master.font_name, 50))
        self.black_conter_label.place(x=10, y=10)

        self.white_conter_label = tk.Label(self, text="W00", fg="#EEEEEE", bg="#808080", font = (master.font_name, 50))
        self.white_conter_label.place(x=530, y=10)

        self.button1 = tk.Button(self, text="Next", font = (master.font_name, 50), command=lambda:self.button1_click())
        self.button1.place(x=750, y=380)
    
    def canvas_update(self, state=0, x=0, y=0, oldcolor=0):
        if state==0:
            self.master.se1.play()
        elif state==1:
            self.master.se2.play()
        elif state==2:
            self.master.se3.play()
        self.stone_counter_update()
        self.game_canvas.delete("all")
        self.game_canvas.configure(bg="#44EE88")
        self.game_canvas.create_rectangle(0, 0, self.canvas_width+10, self.canvas_height+10, fill = "#22FF77")
        for i in range(9):
            self.game_canvas.create_line(10+self.cell_width*i, 10, 10+self.cell_width*i, 10+self.cell_height*8, fill="#101010", width=2)
            self.game_canvas.create_line(10, 10+self.cell_height*i, 10+self.cell_width*8, 10+self.cell_height*i, fill="#101010", width=2)
        for i in range(8):
            for j in range(8):
                t = (j, i)
                if (self.master.board.stone_exist >> (board.Board.t2n(t)) &1)==1 :
                    if (self.master.board.stone_black >> (board.Board.t2n(t)) &1)==1 :
                        self.stone_black_draw(i, j)
                    else:
                        self.stone_white_draw(i, j)
        lp = self.master.board.list_placable()
        if state==0:
            for w in lp:
                i = w%8
                j = w//8
                self.stone_blue_draw(i,j)
        if state==1:
            self.stone_yellow_draw(x,y)
            self.master.after(800, self.canvas_update, 2, x, y, self.master.board.stone_black)
        if state==2:
            for i in range(8):
                for j in range(8):
                    t = (j, i)
                    if (self.master.board.stone_exist >> (board.Board.t2n(t)) &1)==1 :
                        if (self.master.board.stone_black >> (board.Board.t2n(t)) &1) ^ (oldcolor >> (board.Board.t2n(t)) &1)==1 :
                            self.stone_gray_draw(i, j)
            self.stone_yellow_draw(x,y)
            self.master.after(800, self.canvas_update, 0, 0, 0, 0)


    def stone_black_draw(self, x, y):
        self.game_canvas.create_oval(11+self.cell_width*x, 11+self.cell_height*y, 9+self.cell_width*(x+1), 9+self.cell_height*(y+1), fill="#111111")
        for k in range(17):
            k_ = 17 - k
            s = format(3*k, "02X")
            s = "#" + s + s + s
            self.game_canvas.create_oval(10+self.cell_width*(2*x+1)//2 - (k_+1), 10+self.cell_height*(2*y+1)//2  - (k_+1), 10+self.cell_width*(2*x+1)//2  + (k_+1), 10+self.cell_height*(2*y+1)//2   + (k_+1), fill=s, outline = s)

    def stone_white_draw(self, x, y):
        self.game_canvas.create_oval(11+self.cell_width*x, 11+self.cell_height*y, 9+self.cell_width*(x+1), 9+self.cell_height*(y+1), fill="#EEEEEE")
        for k in range(17):
            k_ = 17 - k
            s = format((0xEE-17*3)+3*k_, "02X")
            s = "#" + s + s + s
            self.game_canvas.create_oval(10+self.cell_width*(2*x+1)//2 - (k_+1), 10+self.cell_height*(2*y+1)//2  - (k_+1), 10+self.cell_width*(2*x+1)//2  + (k_+1), 10+self.cell_height*(2*y+1)//2   + (k_+1), fill=s, outline = s)

    def stone_gray_draw(self, x, y):
        self.game_canvas.create_oval(11+self.cell_width*x, 11+self.cell_height*y, 9+self.cell_width*(x+1), 9+self.cell_height*(y+1), fill="#777777")
        for k in range(17):
            k_ = 17 - k
            s = format(0x77+k, "02X")
            s = "#" + s + s + s
            self.game_canvas.create_oval(10+self.cell_width*(2*x+1)//2 - (k_+1), 10+self.cell_height*(2*y+1)//2  - (k_+1), 10+self.cell_width*(2*x+1)//2  + (k_+1), 10+self.cell_height*(2*y+1)//2   + (k_+1), fill=s, outline = s)


    def stone_blue_draw(self, x, y):
        self.game_canvas.create_oval(11+self.cell_width*x, 11+self.cell_height*y, 9+self.cell_width*(x+1), 9+self.cell_height*(y+1), fill="#11FFFF")
        for k in range(17):
            k_ = 17 - k
            s = format(0xff-3*k, "02X")
            s = "#" + "11" + s + s
            self.game_canvas.create_oval(10+self.cell_width*(2*x+1)//2 - (k_+1), 10+self.cell_height*(2*y+1)//2  - (k_+1), 10+self.cell_width*(2*x+1)//2  + (k_+1), 10+self.cell_height*(2*y+1)//2   + (k_+1), fill=s, outline = s)

    def stone_yellow_draw(self, x, y):
        self.game_canvas.create_oval(11+self.cell_width*x, 11+self.cell_height*y, 9+self.cell_width*(x+1), 9+self.cell_height*(y+1), fill="#FFFF44")
        for k in range(17):
            k_ = 17 - k
            s = format(0xff-3*k, "02X")
            s = "#" + s + s + "44"
            self.game_canvas.create_oval(10+self.cell_width*(2*x+1)//2 - (k_+1), 10+self.cell_height*(2*y+1)//2  - (k_+1), 10+self.cell_width*(2*x+1)//2  + (k_+1), 10+self.cell_height*(2*y+1)//2   + (k_+1), fill=s, outline = s)


    def stone_counter_update(self):
        bnum = 0
        wnum = 0
        for i in range(8):
            for j in range(8):
                t = (j, i)
                if (self.master.board.stone_exist >> (board.Board.t2n(t)) &1)==1 :
                    if (self.master.board.stone_black >> (board.Board.t2n(t)) &1)==1 :
                        bnum += 1
                    else:
                        wnum += 1
        self.black_conter_label.configure(text=format(bnum, "02d") )
        self.white_conter_label.configure(text=format(wnum, "02d") )

        self.counter_bar.delete("all")
        bw_bounder_x = int((self.canvas_width+10) * (math.tanh( (bnum/(bnum+wnum)-0.5)*3 )+1) / 2  )
        self.counter_bar.create_rectangle(0, 0, self.canvas_width+10, 100, fill = "#22FF77")
        self.counter_bar.create_rectangle(0, 0, bw_bounder_x, 100, fill = "#000000", outline="#000000")
        for i in range(30):
            s = format((30-i)*4, "02X" )
            s = "#" + s + s + s
            self.counter_bar.create_rectangle(0, i, bw_bounder_x, 1+i, fill = s, outline=s)
        self.counter_bar.create_rectangle(bw_bounder_x, 0, self.canvas_width+10, 100, fill = "#EEEEEE")
        for i in range(30):
            s = format((0xEE-30*3)+i*3, "02X" )
            s = "#" + s + s + s
            self.counter_bar.create_rectangle(bw_bounder_x, i, self.canvas_width+10, 1+i, fill = s, outline=s)
        return
        
    def button1_click(self):
        if self.master.board.turn==1 and self.player1==0:
            print("COMは石を置けなかった(先攻)")
            return
        if self.master.board.turn==1 and self.player1==1:
            print()
        if self.master.board.turn==0 and self.player2==0:
            print("COMは石を置けなかった(後攻)")
            return
        if self.master.board.turn==0 and self.player2==1:
            print()
        self.game_still_cont = self.master.board.can_continue(True)
        self.game_still_cont = self.master.board.can_continue(True)
        if self.game_still_cont:
            n = self.master.board.get_action()
            self.canvas_update(1, n%8, n//8)
            self.master.board.put_stone(n)
            self.game_still_cont = self.master.board.can_continue()
        print("COMが石を置いた")
        self.master.after(1800, self.button1_click)
    
    def human_put_stone(self, x, y):
        if x<0 or y<0 or x>=8 or y>=8:
            return
        t = (y, x)
        if (self.master.board.stone_exist >> board.Board.t2n(t)) & 1:
            return
        if self.master.board.is_placable(board.Board.t2n(t))==False:
            return
        self.canvas_update(1, x, y)
        self.game_still_cont = self.master.board.can_continue(True)
        self.game_still_cont = self.master.board.can_continue(True)
        if self.game_still_cont:
            n = board.Board.t2n(t)
            self.master.board.put_stone(n)
            self.game_still_cont = self.master.board.can_continue()
        self.master.after(1800, self.button1_click)
        return

    def cell_click(self, event):
        #print(self.master.board.turn)
        #print(self.player1)

        if self.master.board.turn==1 and self.player1==0:
            print()
        if self.master.board.turn==1 and self.player1==1:
            print("あなたの番ではありません：")
            return
        if self.master.board.turn==0 and self.player2==0:
            print()
        if self.master.board.turn==0 and self.player2==1:
            print("あなたの番ではありません：")
            return
        x = event.x
        y = event.y
        x = (x-10) // self.cell_width
        y = (y-10) // self.cell_height
        if x<0 or y<0 or x>=8 or y>=8:
            return
        self.human_put_stone(x, y)
        #self.canvas_update()
        print("セルが押された：", x, ",", y)



class ResultPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        pygame.init()
        self.master = master
        self.configure(bg="#992299")
        self.grid(row=0, column=0, sticky="nsew")





class App(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.width = 640
        self.height = 480
        self.font_name = "ヒラギノ"
        self.geometry( str(self.width) + "x" + str(self.height) )
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.start_page = StartPage(self)
        self.option_page = OptionFrame(self)
        self.game_page = GamePage(self)
        self.result_page = ResultPage(self)

        self.board = board.Board()

        sound_folder_path = "../sound/"

        self.bgm1 = pygame.mixer.Sound(sound_folder_path+"maou09.mp3")
        self.bgm1.play(loops=-1)

        self.se1 = pygame.mixer.Sound(sound_folder_path+"maou47.wav")
        self.se2 = pygame.mixer.Sound(sound_folder_path+"maou41.wav")
        self.se3 = pygame.mixer.Sound(sound_folder_path+"maou48.wav")
        
        self.start_page.tkraise()


    def com_random(self, board : board.Board):
        return random.choice(board.list_placable())


if __name__ == "__main__":
    # アプリケーションを開始
    tkapp = App()
    tkapp.title('おせろ')
    tkapp.mainloop()