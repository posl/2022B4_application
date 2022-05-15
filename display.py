import tkinter as tk
import tkinter.ttk as ttk
import math
import random

import pygame

import sound



class Page(tk.Frame):
    def __init__(self, par, board):
        # ページに共通する属性やメソッドなどを記述
        tk.Frame.__init__(self, par)
        self.par = par
        self.board = board

        self.win_width = 640
        self.win_height = 480
        self.font_name = "凸版文久見出しゴシック"
        self.grid(row=0, column=0, sticky="nsew") #正常な画面表示に必要
        



class StartPage(Page):
    def __init__(self, par, board):
        Page.__init__(self, par, board)

        self.configure(bg="#881111")

        self.label = tk.Label(self, text="Othello", font = (self.font_name, 50), fg="#119911", bg="#881111")
        self.label.pack(anchor="center", expand=True)
        #self.label.place(x=200, y=50)
        self.button1 = tk.Button(self, text="Play", font = (self.font_name, 50), command=lambda:self.goto_option_page())
        self.button1.pack(anchor="center", expand=True)
        #self.button1.place(x=200, y=200)
        self.button2 = tk.Button(self, text="Quit", font = (self.font_name, 50), command=lambda:exit())
        self.button2.pack(anchor="center", expand=True)
        #self.button2.place(x=200, y=270)

    def goto_option_page(self):
        self.par.option_page.set_player_kinds()
        self.par.option_page.combobox1_changed()
        self.par.option_page.combobox2_changed()
        self.par.change_page(1)
        


class OptionPage(Page):
    def __init__(self, par, board):
        Page.__init__(self, par, board)
        self.par = par
        self.board = board
        self.configure(bg="#992299")

        self.combo_menus = ["---"]
        self.combo_menus2 = ("x1", "x2", "x3", "x4")

        self.label1 = tk.Label(self, text="Player1", fg="#999999")
        self.label1.place(x=10, y=30)

        self.label2 = tk.Label(self, text="Player2", fg="#999999")
        self.label2.place(x=10, y=90)

        self.label3 = tk.Label(self, text="表示速度", fg="#999999")
        self.label3.place(x=10, y=150)

        self.combobox3 = ttk.Combobox(self, height=3, values = self.combo_menus, state="readonly")
        self.combobox3.place(x=330, y=30 )
        self.combobox3.current(0)

        self.combobox4 = ttk.Combobox(self, height=3, values = self.combo_menus, state="readonly")
        self.combobox4.place(x=330, y=90 )
        self.combobox4.current(0)

        self.combobox1 = ttk.Combobox(self, height=3, values = self.combo_menus, state="readonly")
        self.combobox1.place(x=100, y=30 )
        self.combobox1.current(0)
        self.combobox1.bind("<<ComboboxSelected>>",lambda e: self.combobox1_changed() )
        

        self.combobox2 = ttk.Combobox(self, height=3, values = self.combo_menus, state="readonly")
        self.combobox2.place(x=100, y=90 )
        self.combobox2.current(0)
        self.combobox2.bind("<<ComboboxSelected>>",lambda e: self.combobox2_changed() )

        self.combobox5 = ttk.Combobox(self, height=4, values = self.combo_menus2, state="readonly")
        self.combobox5.place(x=100, y=150 )
        self.combobox5.current(0)

        self.button1 = tk.Button(self, text="Next", font = (self.font_name, 50), command=lambda:self.start_game())
        self.button1.place(x=450, y=380)

    def combobox1_changed(self):
        n = self.combobox1.current()
        n = self.board.player_kinds.get_difficulty(n)
        if n<2:
            self.combobox3.place(x = 1000)
            self.combobox3["values"] = ["1"]
            self.combobox3.current(0)
        else:
            self.combobox3.place(x=330)
            self.combo_menus.clear()
            for i in range(n):
                self.combo_menus.append("難易度"+str(i+1))
            self.combobox3["values"] = self.combo_menus
            self.combobox3.current(0)

    def combobox2_changed(self):
        n = self.combobox2.current()
        n = self.board.player_kinds.get_difficulty(n)
        if n<2:
            self.combobox4.place(x = 1000)
            self.combobox4["values"] = ["1"]
            self.combobox4.current(0)
        else:
            self.combobox4.place(x=330)
            self.combo_menus.clear()
            for i in range(n):
                self.combo_menus.append("難易度"+str(i+1))
            self.combobox4["values"] = self.combo_menus
            self.combobox4.current(0)

    def set_player_kinds(self):
        n = self.board.player_kinds.get_num()
        self.combo_menus.clear()
        for i in range(n):
            self.combo_menus.append(self.board.player_kinds.get_name(i))
        self.combobox1["values"] = self.combo_menus
        self.combobox2["values"] = self.combo_menus
        self.combobox1.current(0)
        self.combobox2.current(0)

    # ゲームの設定を有効化
    def game_config_validate(self):
        player1_id = self.combobox1.current()
        player2_id = self.combobox2.current()
        player1_diff = self.combobox3.current()  #難易度
        player2_diff = self.combobox4.current()  #難易度
        # ボード側に上の値を渡して設定させる処理をここに書く
        self.board.game_config(player1_id, player2_id, player1_diff, player2_diff)

        self.par.game_page.time_len_coef = self.combobox5.current() + 1
        return

    def start_game(self):
        self.game_config_validate()
        self.par.change_page(2)
        self.board.click_attr = True
        self.par.game_page.game_canvas_state = 0
        self.par.quit()





class GamePage(Page):
    def __init__(self, par, board):
        Page.__init__(self, par, board)
        self.configure(bg="#992299")

        self.canvas_width = 400
        self.canvas_height = 400

        self.cell_width = (self.canvas_width-10) // 8
        self.cell_height = (self.canvas_height-10) // 8

        self.player1 = 0
        self.player2 = 0

        self.game_canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height)
        self.game_canvas.place(x=100, y=50)
        self.game_canvas.bind("<Button-1>", self.cell_click)
        self.game_canvas_state = 0
        self.game_canvas_lock = False
        self.time_len_coef = 1

        self.counter_bar = tk.Canvas(self, width=self.canvas_width, height=30)
        self.counter_bar.place(x=100, y=5)

        self.black_conter_label = tk.Label(self, text="B00", fg="#111111", bg="#808080", font = (self.font_name, 50))
        self.black_conter_label.place(x=10, y=10)

        self.white_conter_label = tk.Label(self, text="W00", fg="#EEEEEE", bg="#808080", font = (self.font_name, 50))
        self.white_conter_label.place(x=530, y=10)

        self.label1 = tk.Label(self, text="", fg="#111111", bg="#808080", font = (self.font_name, 25))
        self.label1.place(x=1000, y=300)

        # 見えないところに置かれている、削除するかも？
        self.button1 = tk.Button(self, text="Next", font = (self.font_name, 25), command=lambda:None)
        self.button1.place(x=750, y=380)

        self.button2 = tk.Button(self, text=">", font = (self.font_name, 50), command=lambda:self.goto_start_page())
        self.button2.place(x=750, y=380)
    

    def canvas_update(self, flag=None, n=999):
        print("canvas_update:",self.game_canvas_state)
        self.stone_counter_update()
        if self.game_canvas_state==0:
            self.render_current_board()
            self.game_canvas_state = 1
            self.game_canvas_lock = True
            self.par.after(50//self.time_len_coef, self.canvas_update)
            self.par.mainloop()
        elif self.game_canvas_state==1:
            self.render_placeable()
            self.game_canvas_state = 2
            self.par.after(200//self.time_len_coef, self.canvas_quit)
        elif self.game_canvas_state==2:
            self.render_reverse(n, 0)
            self.game_canvas_state = 3
            self.game_canvas_lock = True
            self.par.after(400//self.time_len_coef, self.canvas_update)
            self.par.mainloop()
        elif self.game_canvas_state==3:
            self.render_reverse(n, 1)
            self.game_canvas_state = 4
            self.par.after(400//self.time_len_coef, self.canvas_update)
        elif self.game_canvas_state==4:
            self.render_reverse(n, 2)
            self.game_canvas_state = 5
            self.par.after(400//self.time_len_coef, self.canvas_update)
        elif self.game_canvas_state==5:
            self.render_current_board()
            self.game_canvas_state = 1
            self.par.after(400//self.time_len_coef, self.canvas_update)

    def canvas_quit(self):
        self.game_canvas_lock = False
        self.par.quit()

    
    def game_canvas_state_update(self, v):
        self.game_canvas_state = v
        
        

    #キャンバスを全消しー＞線描画ー＞黒石、白石の描画
    def render_current_board(self):
        self.par.sounds.play(1)
        self.game_canvas.delete("all")
        self.game_canvas.configure(bg="#44EE88")
        self.game_canvas.create_rectangle(0, 0, self.canvas_width+10, self.canvas_height+10, fill = "#22FF77")
        for i in range(9):
            self.game_canvas.create_line(10+self.cell_width*i, 10, 10+self.cell_width*i, 10+self.cell_height*8, fill="#101010", width=2)
            self.game_canvas.create_line(10, 10+self.cell_height*i, 10+self.cell_width*8, 10+self.cell_height*i, fill="#101010", width=2)
        bplace = self.board.black_positions
        wplace = self.board.white_positions
        for pl in bplace:
            j, i = self.board.n2t(pl)
            self.stone_black_draw(i, j)
        for pl in wplace:
            j, i = self.board.n2t(pl)
            self.stone_white_draw(i, j)
        
    #石が置けるところを青く
    def render_placeable(self):
        #self.par.sounds.play(2)
        lp = self.board.list_placable()
        for w in lp:
            j, i = self.board.n2t(w)
            self.stone_red_draw(i,j)
        return

    # flg...True : ひっくり返るところをgray
    # flg...False : ひっくり返ったところを黒/白に
    def render_reverse(self, n, flg = 0):
        y, x = self.board.n2t(n)
        r_list =  self.board.reverse_positions
        bplace = self.board.black_positions
        wplace = self.board.white_positions
        if flg==0:
            self.par.sounds.play(2)
            self.stone_yellow_draw(x, y)
        elif flg==1:
            self.par.sounds.play(3)
            for i in r_list:
                i_y, i_x = self.board.n2t(i)
                self.stone_gray_draw(i_x, i_y)
            #self.par.after(500, self.render_reverse, False)
        elif flg==2:
            self.par.sounds.play(3)
            for i in r_list:
                i_y, i_x = self.board.n2t(i)
                if i in bplace:
                    self.stone_black_draw(i_x, i_y)
                if i in wplace:
                    self.stone_white_draw(i_x, i_y)
                #self.stone_gray_draw(i_x, i_y)
                #self.par.after(500, self.canvas_update, False)
        return


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
        self.game_canvas.create_rectangle(11+self.cell_width*x, 11+self.cell_height*y, 9+self.cell_width*(x+1), 9+self.cell_height*(y+1), fill="#11FFFF")
        for k in range(17):
            k_ = 17 - k
            s = format(0xff-3*k, "02X")
            s = "#" + "11" + s + s
            self.game_canvas.create_rectangle(10+self.cell_width*(2*x+1)//2 - (k_+1), 10+self.cell_height*(2*y+1)//2  - (k_+1), 10+self.cell_width*(2*x+1)//2  + (k_+1), 10+self.cell_height*(2*y+1)//2   + (k_+1), fill=s, outline = s)

    def stone_red_draw(self, x, y):
        self.game_canvas.create_rectangle(11+self.cell_width*x, 11+self.cell_height*y, 9+self.cell_width*(x+1), 9+self.cell_height*(y+1), fill="#FF3333")
        for k in range(17):
            k_ = 17 - k
            s = format(0xff-3*k, "02X")
            s = "#" + s + "33" + "33"
            self.game_canvas.create_rectangle(10+self.cell_width*(2*x+1)//2 - (k_+1), 10+self.cell_height*(2*y+1)//2  - (k_+1), 10+self.cell_width*(2*x+1)//2  + (k_+1), 10+self.cell_height*(2*y+1)//2   + (k_+1), fill=s, outline = s)

    def stone_yellow_draw(self, x, y):
        self.game_canvas.create_oval(11+self.cell_width*x, 11+self.cell_height*y, 9+self.cell_width*(x+1), 9+self.cell_height*(y+1), fill="#FFFF44")
        for k in range(17):
            k_ = 17 - k
            s = format(0xff-3*k, "02X")
            s = "#" + s + s + "44"
            self.game_canvas.create_oval(10+self.cell_width*(2*x+1)//2 - (k_+1), 10+self.cell_height*(2*y+1)//2  - (k_+1), 10+self.cell_width*(2*x+1)//2  + (k_+1), 10+self.cell_height*(2*y+1)//2   + (k_+1), fill=s, outline = s)


    def stone_counter_update(self):
        bnum = self.board.black_num
        wnum = self.board.white_num

        self.black_conter_label.configure(text=format(bnum, "02d") )
        self.white_conter_label.configure(text=format(wnum, "02d") )

        self.counter_bar.delete("all")
        bw_bounder_x = int((self.canvas_width+10) * (math.tanh( (bnum/(bnum+wnum+0.1)-0.5)*3 )+1) / 2  )
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


    def game_exit_check(self):
        flg = 0
        flg = self.board.can_continue(True)
        flg = self.board.can_continue(True)
        if flg:
            self.board.game_playing = 1
            return
        self.board.game_playing = 0
        self.board.after(1500, self.result_view)

    def cell_click(self, event):
        """
        if self.board.turn==1 and self.player1==0:
            print()
        if self.board.turn==1 and self.player1>0:
            self.board.se4.play()
            print("あなたの番ではありません：")
            return
        if self.board.turn==0 and self.player2==0:
            print()
        if self.board.turn==0 and self.player2>0:
            self.par.play(4)
            print("あなたの番ではありません：")
            return
        """
        if self.game_canvas_lock == True:
            return
        print("cell_click:",self.game_canvas_state)
        x = event.x
        y = event.y
        x = (x-10) // self.cell_width
        y = (y-10) // self.cell_height
        t = (y, x)
        self.board.click_attr = self.board.t2n(t)
        print("セルが押された：", x+1, ",", y+1)
        self.par.after(100, self.quit)
        #self.board.player_action = n

    def result_view(self):
        self.win_check()
        self.button2.place(x=520)
        self.label1.place(x=520)

    def goto_start_page(self):
        self.par.change_page(0)
        self.button2.place(x=1000)
        self.label1.place(x=1000)
        self.label1.configure(text="")
        self.par.quit()
    

    def win_check(self):
        bnum = self.board.black_num
        wnum = self.board.white_num
        if bnum>wnum:
            self.label1.configure(text="黒の勝ち")
        elif bnum<wnum:
            self.label1.configure(text="白の勝ち")
        else:
            self.label1.configure(text="引き分け")




class MainWindow(tk.Tk):
    def __init__(self, board, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        pygame.init()
        self.board = board
        self.width = 640
        self.height = 480
        self.geometry( str(self.width) + "x" + str(self.height) )
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.start_page = StartPage(self, self.board)
        self.option_page = OptionPage(self, self.board)
        self.game_page = GamePage(self, self.board)
        self.change_page(0)

        self.sounds = sound.Sounds()

    def change_page(self, page_id):
        if page_id==0:
            self.start_page.tkraise()
        elif page_id==1:
            self.option_page.tkraise()
        elif page_id==2:
            self.game_page.tkraise()

    




if __name__ == "__main__":
    pass
    # アプリケーションを開始
    #board = board.Board()
    #board.play()
    
    #tkapp = App()
    #tkapp.title('おせろ')
    
    #tkapp.mainloop()
    #board.set_display(tkapp)
