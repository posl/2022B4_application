import os
import tkinter as tk
import tkinter.ttk as ttk
import math
from random import choice, randrange

# pygame のウェルカムメッセージを表示させないための設定
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

from mc_tree_search import MonteCarloTreeSearch, RootPalallelMonteCarloTreeSearch
from mc_primitive import PrimitiveMonteCarlo, NAPrimitiveMonteCarlo
from gt_alpha_beta import AlphaBeta
from drl_rainbow import RainbowComputer
from drl_reinforce import ReinforceComputer
from drl_alphazero import AlphaZeroComputer

from gui_network import NetrorkPlayer
from board import Board
from pyx.speedup import get_stand_bits



class Page(tk.Frame):
    def __init__(self, par, board):
        # ページに共通する属性やメソッドなどを記述
        tk.Frame.__init__(self, par)
        self.par = par
        self.board = board

        self.win_width = 640*2
        self.win_height = 480*2
        self.font_name = "凸版文久見出しゴシック"
        self.grid(row=0, column=0, sticky="nsew") #正常な画面表示に必要




class StartPage(Page):
    def __init__(self, par, board):
        Page.__init__(self, par, board)

        self.configure(bg="#881111")

        self.label = tk.Label(self, text="Othello", font = (self.font_name, 100), fg="#119911", bg="#881111")
        self.label.pack(anchor="center", expand=True)
        #self.label.place(x=200, y=50)
        self.button1 = tk.Button(self, text="Play", font = (self.font_name, 100), command=lambda:self.goto_option_page())
        self.button1.pack(anchor="center", expand=True)
        #self.button1.place(x=200, y=200)
        self.button2 = tk.Button(self, text="Quit", font = (self.font_name, 100), command=lambda:exit())
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

        self.label1 = tk.Label(self, text="Player1", fg="#999999", font = (self.font_name, 30))
        self.label1.place(x=10, y=30)

        self.label2 = tk.Label(self, text="Player2", fg="#999999", font = (self.font_name, 30))
        self.label2.place(x=10, y=200)

        self.label3 = tk.Label(self, text="表示速度", fg="#999999", font = (self.font_name, 30))
        self.label3.place(x=10, y=370)

        self.combobox3 = ttk.Combobox(self, height=4, font = (self.font_name, 30), values = self.combo_menus, state="readonly")
        self.combobox3.place(x=630, y=30 )
        self.combobox3.current(0)

        self.combobox4 = ttk.Combobox(self, height=4, font = (self.font_name, 30), values = self.combo_menus, state="readonly")
        self.combobox4.place(x=630, y=200 )
        self.combobox4.current(0)

        self.combobox1 = ttk.Combobox(self, height=8, font = (self.font_name, 30), values = self.combo_menus, state="readonly")
        self.combobox1.place(x=200, y=30 )
        self.combobox1.current(0)
        self.combobox1.bind("<<ComboboxSelected>>",lambda e: self.combobox1_changed() )

        self.combobox2 = ttk.Combobox(self, height=8, font = (self.font_name, 30), values = self.combo_menus, state="readonly")
        self.combobox2.place(x=200, y=200 )
        self.combobox2.current(0)
        self.combobox2.bind("<<ComboboxSelected>>",lambda e: self.combobox2_changed() )

        self.combobox5 = ttk.Combobox(self, height=4, font = (self.font_name, 30), values = self.combo_menus2, state="readonly")
        self.combobox5.place(x=200, y=370 )
        self.combobox5.current(0)

        self.button1 = tk.Button(self, text="Next", font = (self.font_name, 100), command=lambda:self.start_game())
        self.button1.place(x=900, y=2*350)

    def combobox1_changed(self):
        n = self.combobox1.current()
        n = self.board.player_kinds.get_difficulty(n)
        if n<2:
            self.combobox3.place(x = 3000)
            self.combobox3["values"] = ["1"]
            self.combobox3.current(0)
        else:
            self.combobox3.place(x=630)
            self.combo_menus.clear()
            for i in range(n):
                self.combo_menus.append("難易度"+str(i+1))
            self.combobox3["values"] = self.combo_menus
            self.combobox3.current(0)

    def combobox2_changed(self):
        n = self.combobox2.current()
        n = self.board.player_kinds.get_difficulty(n)
        if n<2:
            self.combobox4.place(x = 3000)
            self.combobox4["values"] = ["1"]
            self.combobox4.current(0)
        else:
            self.combobox4.place(x=630)
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

        name1 = self.board.player_kinds.get_name(player1_id)
        name2 = self.board.player_kinds.get_name(player2_id)
        self.par.title(name1 + "Lv." + str(player1_diff+1) + "(黒) vs " + name2  + "Lv." + str(player2_diff+1) + "(白)")
        return

    def start_game(self):
        self.game_config_validate()
        self.par.change_page(2)
        self.board.click_attr = True
        self.par.game_page.game_canvas_state = 0
        self.par.sounds.bgm_play(1)
        self.par.quit()




class GamePage(Page):
    def __init__(self, par, board):
        Page.__init__(self, par, board)
        self.configure(bg="#992299")

        self.canvas_width = 15 + 2*45*8
        self.canvas_height = 15 + 2*45*8

        self.cell_width = (self.canvas_width-15) // 8
        self.cell_height = (self.canvas_height-15) // 8

        self.player1 = 0
        self.player2 = 0

        self.game_canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height)
        self.game_canvas.place(x=200, y=100)
        self.game_canvas.bind("<Button-1>", self.cell_click)
        self.game_canvas_state = 0
        self.game_canvas_lock = False
        self.time_len_coef = 1

        self.counter_bar = tk.Canvas(self, width=self.canvas_width, height=60)
        self.counter_bar.place(x=200, y=5)

        self.black_conter_label = tk.Label(self, text="B00", fg="#111111", bg="#808080", font = (self.font_name, 90))
        self.black_conter_label.place(x=10, y=10)

        self.white_conter_label = tk.Label(self, text="W00", fg="#EEEEEE", bg="#808080", font = (self.font_name, 90))
        self.white_conter_label.place(x=10 + self.canvas_width + 300, y=10)

        self.label1 = tk.Label(self, text="", fg="#111111", bg="#808080", font = (self.font_name, 50))
        self.label1.place(x=3000, y=270*2)

        # 見えないところに置かれている、削除するかも？
        self.button1 = tk.Button(self, text="Next", font = (self.font_name, 25), command=lambda:None)
        self.button1.place(x=3000, y=380)

        self.button2 = tk.Button(self, text=">", font = (self.font_name, 70), command=lambda:self.goto_result_page())
        self.button2.place(x=3000, y=340*2)

        self.button3 = tk.Button(self, text="X", font = (self.font_name, 50), command=lambda:self.goto_start_page())
        self.button3.place(x=3000, y=160)

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
        for k in range(self.cell_width//2-3):
            k_ = self.cell_width//2-3 - k
            s = format(3*k, "02X")
            s = "#" + s + s + s
            self.game_canvas.create_oval(10+self.cell_width*(2*x+1)//2 - (k_+1), 10+self.cell_height*(2*y+1)//2  - (k_+1), 10+self.cell_width*(2*x+1)//2  + (k_+1), 10+self.cell_height*(2*y+1)//2   + (k_+1), fill=s, outline = s)

    def stone_white_draw(self, x, y):
        self.game_canvas.create_oval(11+self.cell_width*x, 11+self.cell_height*y, 9+self.cell_width*(x+1), 9+self.cell_height*(y+1), fill="#EEEEEE")
        for k in range(self.cell_width//2-3):
            k_ = self.cell_width//2-3 - k
            s = format((0xEE-(self.cell_width//2-3)*3)+3*k_, "02X")
            s = "#" + s + s + s
            self.game_canvas.create_oval(10+self.cell_width*(2*x+1)//2 - (k_+1), 10+self.cell_height*(2*y+1)//2  - (k_+1), 10+self.cell_width*(2*x+1)//2  + (k_+1), 10+self.cell_height*(2*y+1)//2   + (k_+1), fill=s, outline = s)

    def stone_gray_draw(self, x, y):
        self.game_canvas.create_oval(11+self.cell_width*x, 11+self.cell_height*y, 9+self.cell_width*(x+1), 9+self.cell_height*(y+1), fill="#777777")
        for k in range(self.cell_width//2-3):
            k_ = self.cell_width//2-3 - k
            s = format(0x77+k, "02X")
            s = "#" + s + s + s
            self.game_canvas.create_oval(10+self.cell_width*(2*x+1)//2 - (k_+1), 10+self.cell_height*(2*y+1)//2  - (k_+1), 10+self.cell_width*(2*x+1)//2  + (k_+1), 10+self.cell_height*(2*y+1)//2   + (k_+1), fill=s, outline = s)


    def stone_blue_draw(self, x, y):
        self.game_canvas.create_rectangle(11+self.cell_width*x, 11+self.cell_height*y, 9+self.cell_width*(x+1), 9+self.cell_height*(y+1), fill="#11FFFF")
        for k in range(self.cell_width//2-3):
            k_ = self.cell_width//2-3 - k
            s = format(0xff-3*k, "02X")
            s = "#" + "11" + s + s
            self.game_canvas.create_rectangle(10+self.cell_width*(2*x+1)//2 - (k_+1), 10+self.cell_height*(2*y+1)//2  - (k_+1), 10+self.cell_width*(2*x+1)//2  + (k_+1), 10+self.cell_height*(2*y+1)//2   + (k_+1), fill=s, outline = s)

    def stone_red_draw(self, x, y):
        self.game_canvas.create_rectangle(11+self.cell_width*x, 11+self.cell_height*y, 9+self.cell_width*(x+1), 9+self.cell_height*(y+1), fill="#FF3333")
        for k in range(self.cell_width//2-3):
            k_ = self.cell_width//2-3 - k
            s = format(0xff-3*k, "02X")
            s = "#" + s + "33" + "33"
            self.game_canvas.create_rectangle(10+self.cell_width*(2*x+1)//2 - (k_+1), 10+self.cell_height*(2*y+1)//2  - (k_+1), 10+self.cell_width*(2*x+1)//2  + (k_+1), 10+self.cell_height*(2*y+1)//2   + (k_+1), fill=s, outline = s)

    def stone_yellow_draw(self, x, y):
        self.game_canvas.create_oval(11+self.cell_width*x, 11+self.cell_height*y, 9+self.cell_width*(x+1), 9+self.cell_height*(y+1), fill="#FFFF44")
        for k in range(self.cell_width//2-3):
            k_ = self.cell_width//2-3 - k
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
        for i in range(60):
            s = format((60-i)*2, "02X" )
            s = "#" + s + s + s
            self.counter_bar.create_rectangle(0, i, bw_bounder_x, 1+i, fill = s, outline=s)
        self.counter_bar.create_rectangle(bw_bounder_x, 0, self.canvas_width+10, 100, fill = "#EEEEEE")
        for i in range(60):
            s = format((0xEE-60*1)+i*1, "02X" )
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
        self.button2.place(x=1020)
        self.label1.place(x=1020)

    def goto_start_page(self):
        self.par.change_page(0)
        self.button2.place(x=3000)
        self.label1.place(x=3000)
        self.label1.configure(text="")
        self.par.sounds.bgm_play(0)
        self.par.quit()

    def goto_result_page(self):
        self.par.result_page.graph_draw()
        self.par.change_page(3)
        self.button2.place(x=3000)
        self.label1.place(x=3000)
        self.label1.configure(text="")
        print(self.board.play_log)

    def win_check(self):
        bnum = self.board.black_num
        wnum = self.board.white_num
        if bnum>wnum:
            self.label1.configure(text="黒の勝ち")
        elif bnum<wnum:
            self.label1.configure(text="白の勝ち")
        else:
            self.label1.configure(text="引き分け")




class ResultPage(Page):
    def __init__(self, par, board):
        Page.__init__(self, par, board)
        self.configure(bg="#992299")

        self.canvas_width = 400
        self.canvas_height = 400

        self.cell_width = (self.canvas_width-10) // 8
        self.cell_height = (self.canvas_height-10) // 8

        self.board_canvas_width = 10 + 120*2
        self.board_canvas_height = 10 + 120*2

        self.cur = 0

        self.stonenum_canvas_width = 600*2
        self.stonenum_canvas_height = 280*2

        self.board_canvas = tk.Canvas(self, width=self.board_canvas_width, height=self.board_canvas_height)
        self.board_canvas.place(x=400, y=10)

        self.stonenum_canvas = tk.Canvas(self, width=self.stonenum_canvas_width, height=self.stonenum_canvas_height)
        self.stonenum_canvas.place(x=10, y=280)
        self.stonenum_canvas.bind("<Button-1>", self.graph_click)
        self.curline1 = None
        self.curline2 = None
        self.curline3 = None
        self.curline4 = None

        self.button1 = tk.Button(self, text="開始画面へ", font = (self.font_name, 50), command=lambda:self.goto_start_page())
        self.button1.place(x=1000, y=10)

        self.button2 = tk.Button(self, width=1, height=3, text=">", font = (self.font_name, 30), command=lambda:self.cur_inc())
        self.button2.place(x=700, y=30)

        self.button3 = tk.Button(self, width=1, height=3, text="<", font = (self.font_name, 30), command=lambda:self.cur_dec())
        self.button3.place(x=280, y=30)

    def graph_click(self, event):
        states = self.board.play_log
        num = len(states)
        turn_width = 1.0 * self.stonenum_canvas_width / num
        x = event.x
        x = int(x/turn_width)
        if x < 0 : x = 0
        if x >= num: x = num - 1
        self.cur = x
        self.miniboard_draw()
        return

    def graph_draw(self):
        self.cur = 0
        states = self.board.play_log
        num = len(states)
        turn_width = 1.0 * self.stonenum_canvas_width / num
        turn_height = 1.0 * self.stonenum_canvas_height / 64
        self.stonenum_canvas.delete("all")
        self.stonenum_canvas.create_rectangle(0, 0, self.stonenum_canvas_width, self.stonenum_canvas_height, fill="#55FF88")
        for i in range(num):
            state = states[i]
            state_b = state[0]
            state_w = state[1]
            bnum = bin(state_b).count("1")
            wnum = bin(state_w).count("1")
            left_x = float(turn_width*i+1)
            right_x = float(turn_width*(i+1)-1)
            bound_bg_y = float(turn_height*bnum) 
            bound_wg_y = float(turn_height*(64-wnum))
            self.stonenum_canvas.create_rectangle(left_x, 0, right_x, bound_bg_y, fill="#111111")
            self.stonenum_canvas.create_rectangle(left_x, bound_wg_y, right_x, self.stonenum_canvas_height, fill="#EEEEEE")
        self.stonenum_canvas.create_line(0, self.stonenum_canvas_height//2, self.stonenum_canvas_width, self.stonenum_canvas_height//2, fill="#FF0000")
        self.curline1 = self.stonenum_canvas.create_line(-10, 0, -10, self.stonenum_canvas_height, fill="#0000FF")
        self.curline2 = self.stonenum_canvas.create_line(0, -10, turn_width-2, -10, fill="#0000FF")
        self.curline3 = self.stonenum_canvas.create_line(-10, 0, -10, self.stonenum_canvas_height, fill="#0000FF")
        self.curline4 = self.stonenum_canvas.create_line(0, -10, turn_width-2, -10, fill="#0000FF")
        self.miniboard_draw()

    def curdraw(self):
        states = self.board.play_log
        num = len(states)
        turn_width = 1.0 * self.stonenum_canvas_width / num
        turn_height = 1.0 * self.stonenum_canvas_height / 64
        self.stonenum_canvas.moveto(self.curline1, turn_width*self.cur-2, 0)  # 左上ー左下
        self.stonenum_canvas.moveto(self.curline2, turn_width*self.cur-2, 0)  # 左上ー右上
        self.stonenum_canvas.moveto(self.curline3, turn_width*(self.cur+1)-2, 0) # 右上ー右下
        self.stonenum_canvas.moveto(self.curline4, turn_width*self.cur-2, self.stonenum_canvas_height)  # 左下ー右下
        pass

    def miniboard_draw(self):
        self.curdraw()
        states = self.board.play_log
        state = states[self.cur]
        self.board_canvas.delete("all")
        self.board_canvas.create_rectangle(0, 0, self.board_canvas_width, self.board_canvas_height, fill="#55FF88")
        for i in range(9):
            self.board_canvas.create_line(5+30*i, 5, 5+30*i, 5+30*8, fill="#000000")
            self.board_canvas.create_line(5, 5+30*i, 5+30*8, 5+30*i, fill="#000000")
        for i in range(8):
            for j in range(8):
                t = (j, i)
                n = self.board.t2n(t)
                if (state[0]>>n)&1:
                    self.draw_stone("#111111", i, j)
                elif (state[1]>>n)&1:
                    self.draw_stone("#EEEEEE", i, j)

    def draw_stone(self, color, x, y):
        self.board_canvas.create_rectangle(6+30*x, 6+30*y, 4+30*(x+1), 4+30*(y+1), fill=color, outline="#888888")

    def cur_inc(self):
        self.cur += 1
        if self.cur >= len(self.board.play_log):
            self.cur = len(self.board.play_log)-1
        self.miniboard_draw()

    def cur_dec(self):
        self.cur -= 1
        if self.cur <0:
            self.cur = 0
        self.miniboard_draw()

    def goto_start_page(self):
        self.par.change_page(0)
        self.par.sounds.bgm_play(0)
        self.par.quit()




class Sounds:
    def __init__(self):
        sound_folder_path = os.path.join(os.path.dirname(__file__), "..", "sound", "{}")
        self.sounds = [None]

        for index in (47, 41, 48, 19):
            se = pygame.mixer.Sound(sound_folder_path.format(f"maou{index}.wav"))
            se.set_volume(0.3)
            self.sounds.append(se)

        self.musics = []
        self.musics.append(sound_folder_path.format("maou09.mp3"))
        self.musics.append(sound_folder_path.format("temsu08.mp3"))

        self.bgm_play(0)

    def bgm_play(self, id, loop=-1):
        if id<0 or id>=len(self.musics):
            return
        pygame.mixer.music.load(self.musics[id])
        pygame.mixer.music.play(loop)

    def play(self, id, loop=0):
        if id<0 or id>=len(self.sounds):
            return
        self.sounds[id].play(loops=loop)

    def stop(self, id):
        if id<0 or id>=len(self.sounds):
            return
        self.sounds[id].stop()




class Human:
    par = None
    network_player = None

    def player(self, board):
        placable = set(board.list_placable())
        while True:
            self.par.mainloop()
            n = board.click_attr
            board.click_attr = None
            if n in placable:
                break
        if not self.network_player.NoNetwork:
            self.network_player.notice(n)
        return n

    def __call__(self, board):
        return self.player(board)

    def cheat_player(self, board):
        t = board.turn
        n = 0
        while True:
            self.par.mainloop()
            n = board.click_attr
            board.click_attr = None
            break
        if t==1:
            if ((board.stone_white>>n) & 1):
                board.stone_white = board.stone_white ^ (1<<n)
            board.stone_black = board.stone_black ^ (1<<n)
        else:
            if ((board.stone_black>>n) & 1):
                board.stone_black = board.stone_black ^ (1<<n)
            board.stone_white = board.stone_white ^ (1<<n)
        if ((board.stone_black>>n) & 1) & ((board.stone_white>>n) & 1):
            board.stone_black = board.stone_black ^ (1<<n)
            board.stone_white = board.stone_white ^ (1<<n)
        return self.player(board)

    #本来はここに書くべきではなかろうが暫定的に
    def com_random(self, board):
        return choice(board.list_placable())

    def com_cheater1(self, board):
        t = board.turn
        bplace = board.black_positions
        wplace = board.white_positions
        ret = choice(board.list_placable())
        if len(bplace)+len(wplace)>4:
            if t==1:
                p = choice(wplace)
                if p!=ret:
                    board.stone_white = board.stone_white ^ (1<<p)
                    board.stone_black = board.stone_black ^ (1<<p)
            else:
                p = choice(bplace)
                if p!=ret:
                    board.stone_black = board.stone_black ^ (1<<p)
                    board.stone_white = board.stone_white ^ (1<<p)
        return ret

class ComRand:
    def __call__(self, board):
        return choice(board.list_placable())



class PlayerKinds:
    def __init__(self, par, ip):
        self.kinds_name = [] # 名前（人間、ランダムなど）
        self.kinds_difficulty = [] # 難易度がいくつあるか(0からN-1) １以下なら難易度選択が非表示

        self.kinds_class = [] # クラスを入れる
        self.kinds_args = [] # クラスのinitの引数

        self.kinds_reset = []  # reset関数を呼ぶ必要があるか
        self.kinds_resetargs = [] # reset関数に渡す引数(ID, 難易度)ごとに設定

        NetrorkPlayer.ip = ip
        Human.par = par
        Human.network_player = NetrorkPlayer()
        Human.network_player.reset()

        self.kinds_name.append("人間")
        self.kinds_difficulty.append(1)
        self.kinds_class.append(Human)
        self.kinds_args.append([ () ])
        self.kinds_reset.append(False)
        self.kinds_resetargs.append(None)

        self.kinds_name.append("通信")
        self.kinds_difficulty.append(1)
        self.kinds_class.append(NetrorkPlayer)
        self.kinds_args.append([ () ])
        self.kinds_reset.append(True)
        self.kinds_resetargs.append([ () ])

        self.kinds_name.append("ランダム")
        self.kinds_difficulty.append(1)
        self.kinds_class.append( ComRand )
        self.kinds_args.append( [ () ] )
        self.kinds_reset.append(False)
        self.kinds_resetargs.append(None)

        self.kinds_name.append("MC木探索")
        self.kinds_difficulty.append(4)
        self.kinds_class.append( MonteCarloTreeSearch )
        self.kinds_args.append( [ (1*1024, ), (4*1024, ), (16*1024, ), (64*1024, ) ] )
        self.kinds_reset.append(True)
        self.kinds_resetargs.append( [ (), (), (), ()  ] )

        self.kinds_name.append("MC木探索 + ルート並列化")
        self.kinds_difficulty.append(2)
        self.kinds_class.append( RootPalallelMonteCarloTreeSearch )
        self.kinds_args.append( [ (30000, ), (50000, ) ] )
        self.kinds_reset.append( False )
        self.kinds_resetargs.append( None )

        self.kinds_name.append("原始MC法")
        self.kinds_difficulty.append(4)
        self.kinds_class.append( PrimitiveMonteCarlo )
        self.kinds_args.append( [ (1*256, ), (4*256, ), (16*256, ), (64*256, ) ] )
        self.kinds_reset.append(False)
        self.kinds_resetargs.append( None )

        self.kinds_name.append("原始MC法 + NegaAlpha")
        self.kinds_difficulty.append(4)
        self.kinds_class.append( NAPrimitiveMonteCarlo )
        self.kinds_args.append( [ (1*256, 2), (4*256, 4), (16*256, 8), (32*256, 16) ] )
        self.kinds_reset.append(False)
        self.kinds_resetargs.append( None )

        self.kinds_name.append("AlphaBeta")
        self.kinds_difficulty.append(2)
        self.kinds_class.append( AlphaBeta )
        self.kinds_args.append( [ (0, ), (1, ) ] )
        self.kinds_reset.append(True)
        self.kinds_resetargs.append( [ (), ()  ] )

        self.kinds_name.append("Reinforce")
        self.kinds_difficulty.append(1)
        self.kinds_class.append( ReinforceComputer )
        self.kinds_args.append( [ (64, ) ] )
        self.kinds_reset.append(True)
        self.kinds_resetargs.append( [ ()  ] )

        self.kinds_name.append("RainBow")
        self.kinds_difficulty.append(1)
        self.kinds_class.append( RainbowComputer )
        self.kinds_args.append( [ (64, ) ] )
        self.kinds_reset.append(True)
        self.kinds_resetargs.append( [ ()  ] )

        self.kinds_name.append("AlphaZero")
        self.kinds_difficulty.append(3)
        self.kinds_class.append( AlphaZeroComputer )
        self.kinds_args.append( [ (64, ), (64, ), (64, ) ] )
        self.kinds_reset.append(True)
        self.kinds_resetargs.append( [ (randrange(10), 50), (randrange(5, 10), 200), (None, 800)  ] )


    def get_agent(self, id, diff):
        if id>=0:
            agent = self.kinds_class[id]( * self.kinds_args[id][diff] )
            if self.kinds_reset[id]:
                agent.reset( * self.kinds_resetargs[id][diff] )
            return agent
        else:
            print("Index Error")
            exit()

    def get_num(self):
        return len(self.kinds_name)

    def get_name(self, id):
        if id<0 or id>=len(self.kinds_name):
            print("範囲外のIDが指定されました")
            exit()
        return self.kinds_name[id]

    def get_difficulty(self, id):
        if id<0 or id>=len(self.kinds_difficulty):
            print("範囲外のIDが指定されました")
            exit()
        return self.kinds_difficulty[id]




class MainWindow(tk.Tk):
    def __init__(self, board, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        pygame.init()
        self.board = board
        self.width = 640*2
        self.height = 480*2
        self.geometry( str(self.width) + "x" + str(self.height) )
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.start_page = StartPage(self, self.board)
        self.option_page = OptionPage(self, self.board)
        self.game_page = GamePage(self, self.board)
        self.result_page = ResultPage(self, self.board)
        self.change_page(0)

        self.sounds = Sounds()

        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    def on_exit(self):
        self.destroy()
        exit()

    def change_page(self, page_id):
        if page_id==0:
            self.start_page.tkraise()
        elif page_id==1:
            self.option_page.tkraise()
        elif page_id==2:
            self.game_page.tkraise()
        elif page_id==3:
            self.result_page.tkraise()





class DisplayBoard(Board):
    def __init__(self):
        super().__init__()

        # 画面表示用のクリックイベントを保持するための属性
        self.click_attr = None

        # 画面表示用にどこがひっくり返されたかを保持するための属性
        self.reversed = 0

        # mcのときの不具合を避けるためlog_stateとわける
        self.play_log = []


    def add_playlog(self):
        self.play_log.append(self.state)

    def clear_playlog(self):
        self.play_log.clear()


    @property
    def black_positions(self):
        return get_stand_bits(self.stone_black)

    @property
    def white_positions(self):
        return get_stand_bits(self.stone_white)

    @property
    def reverse_positions(self):
        return get_stand_bits(self.reversed)


    # id...種類のID  diff...難易度
    def game_config(self, player1id, player2id, player1diff=0, player2diff=0):
        agent1 = self.player_kinds.get_agent(player1id, player1diff)
        agent2 = self.player_kinds.get_agent(player2id, player2diff)
        self.set_plan(agent1, agent2)

    def render(self, mask, flag, n = 999):
        self.add_playlog()
        self.reversed = mask
        self.main_window.game_page.canvas_update(flag, n)


    def play(self):
        # ウインドウ
        self.main_window = MainWindow(self)

        # プレイヤーの種類
        self.player_kinds = PlayerKinds(self.main_window, ip)

        while True:
            self.main_window.change_page(0)
            self.main_window.mainloop()
            if self.click_attr:
                self.__play()
            else:
                break
            self.main_window.game_page.result_view()
            self.main_window.mainloop()

    def __play(self):
        # 最初の盤面表示
        self.reset()
        self.print_state()
        self.render(None, None)
        self.main_window.after(100, self.main_window.quit)
        self.main_window.mainloop()

        self.clear_playlog()
        self.add_playlog()
        self.game(self.print_state)

        # 最後の１石だけ表示されない問題を解消する (１秒待機)
        self.main_window.after(1000, self.main_window.quit)
        self.main_window.mainloop()




if __name__ == "__main__":
    ip = input("サーバーのIPを入力してください-ない場合は0に\n")
    displayboard = DisplayBoard()
    displayboard.play(ip)