import os
import tkinter as tk
import tkinter.ttk as ttk
import math
import random

# pygame のウェルカムメッセージを表示させないための設定
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

from mc_tree_search import MonteCarloTreeSearch
from mc_primitive import PrimitiveMonteCarlo, NAPrimitiveMonteCarlo
from gt_alpha_beta import AlphaBeta
from drl_rainbow import RainbowComputer
from drl_reinforce import ReinforceComputer
from drl_alphazero import AlphaZeroComputer

from board import Board
from speedup import get_stand_bits
from network import NetrorkPlayer

IPADDR = "127.0.0.1"



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

        self.combobox3 = ttk.Combobox(self, height=4, values = self.combo_menus, state="readonly")
        self.combobox3.place(x=330, y=30 )
        self.combobox3.current(0)

        self.combobox4 = ttk.Combobox(self, height=4, values = self.combo_menus, state="readonly")
        self.combobox4.place(x=330, y=90 )
        self.combobox4.current(0)

        self.combobox1 = ttk.Combobox(self, height=8, values = self.combo_menus, state="readonly")
        self.combobox1.place(x=100, y=30 )
        self.combobox1.current(0)
        self.combobox1.bind("<<ComboboxSelected>>",lambda e: self.combobox1_changed() )

        self.combobox2 = ttk.Combobox(self, height=8, values = self.combo_menus, state="readonly")
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

        self.button2 = tk.Button(self, text=">", font = (self.font_name, 50), command=lambda:self.goto_result_page())
        self.button2.place(x=750, y=380)

        self.button3 = tk.Button(self, text="X", font = (self.font_name, 50), command=lambda:self.goto_start_page())
        self.button3.place(x=750, y=160)

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
        self.par.sounds.bgm_play(0)
        self.par.quit()

    def goto_result_page(self):
        self.par.result_page.graph_draw()
        self.par.change_page(3)
        self.button2.place(x=1000)
        self.label1.place(x=1000)
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

        self.board_canvas_width = 130
        self.board_canvas_height = 130

        self.cur = 0

        self.stonenum_canvas_width = 600
        self.stonenum_canvas_height = 300

        self.board_canvas = tk.Canvas(self, width=self.board_canvas_width, height=self.board_canvas_height)
        self.board_canvas.place(x=200, y=10)

        self.stonenum_canvas = tk.Canvas(self, width=self.stonenum_canvas_width, height=self.stonenum_canvas_height)
        self.stonenum_canvas.place(x=10, y=150)
        self.stonenum_canvas.bind("<Button-1>", self.graph_click)
        self.curline1 = None
        self.curline2 = None
        self.curline3 = None
        self.curline4 = None

        self.button1 = tk.Button(self, text="開始画面へ", font = (self.font_name, 25), command=lambda:self.goto_start_page())
        self.button1.place(x=480, y=10)

        self.button2 = tk.Button(self, width=1, height=3, text=">", font = (self.font_name, 15), command=lambda:self.cur_inc())
        self.button2.place(x=350, y=30)

        self.button3 = tk.Button(self, width=1, height=3, text="<", font = (self.font_name, 15), command=lambda:self.cur_dec())
        self.button3.place(x=140, y=30)

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
            self.board_canvas.create_line(5+15*i, 5, 5+15*i, 5+15*8, fill="#000000")
            self.board_canvas.create_line(5, 5+15*i, 5+15*8, 5+15*i, fill="#000000")
        for i in range(8):
            for j in range(8):
                t = (j, i)
                n = self.board.t2n(t)
                if (state[0]>>n)&1:
                    self.draw_stone("#111111", i, j)
                elif (state[1]>>n)&1:
                    self.draw_stone("#EEEEEE", i, j)

    def draw_stone(self, color, x, y):
        self.board_canvas.create_rectangle(6+15*x, 6+15*y, 4+15*(x+1), 4+15*(y+1), fill=color, outline="#888888")

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
        self.sounds = []
        self.musics = []
        sound_folder_path = os.path.normpath(os.path.join(os.path.abspath(__file__),  "../sound"))
        se0 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou09.mp3"))
        se1 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou47.wav"))
        se2 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou41.wav"))
        se3 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou48.wav"))
        se4 = pygame.mixer.Sound(os.path.join(sound_folder_path, "maou19.wav"))
        se0.set_volume(0.3)
        se1.set_volume(0.3)
        se2.set_volume(0.3)
        se3.set_volume(0.3)
        se4.set_volume(0.3)

        self.sounds.append(se0)
        self.sounds.append(se1)
        self.sounds.append(se2)
        self.sounds.append(se3)
        self.sounds.append(se4)

        bgm1 = os.path.join(sound_folder_path, "maou09.mp3")
        bgm2 = os.path.join(sound_folder_path, "tamsu08.mp3")

        self.musics.append(bgm1)
        self.musics.append(bgm2)

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
    def __init__(self, par, network_player):
        self.par = par  # par : MainWindow , main_loopを呼び出すために必要
        self.network_player = network_player

    def player(self, board):
        placable = set(board.list_placable())
        while True:
            self.par.mainloop()
            n = board.click_attr
            board.click_attr = None
            if n in placable:
                break
        if self.network_player.is_use:
            self.network_player.notice(n)
        return n

    def cheat_player(self, board):
        t = board.turn
        bplace = board.black_positions
        wplace = board.white_positions
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
        return random.choice(board.list_placable())

    def com_cheater1(self, board):
        t = board.turn
        bplace = board.black_positions
        wplace = board.white_positions
        ret = random.choice(board.list_placable())
        if len(bplace)+len(wplace)>4:
            if t==1:
                p = random.choice(wplace)
                if p!=ret:
                    board.stone_white = board.stone_white ^ (1<<p)
                    board.stone_black = board.stone_black ^ (1<<p)
            else:
                p = random.choice(bplace)
                if p!=ret:
                    board.stone_black = board.stone_black ^ (1<<p)
                    board.stone_white = board.stone_white ^ (1<<p)
        return ret




class PlayerKinds:
    def __init__(self, par):
        self.kinds_name = [] # 名前（人間、ランダムなど）
        self.kinds_func = [] # どこに打つかを返す関数
        self.kinds_difficulty = [] # 難易度がいくつあるか(0からN-1) １以下なら難易度選択が非表示

        self.network_player = NetrorkPlayer(IPADDR)

        self.human = Human(par, self.network_player)
        self.kinds_name.append("人間")
        self.kinds_func.append([self.human.player])
        self.kinds_difficulty.append(1)

        self.kinds_name.append("通信")
        self.kinds_func.append([self.network_player.next_action])
        self.kinds_difficulty.append(1)

        self.kinds_name.append("人間-チート")
        self.kinds_func.append([self.human.cheat_player])
        self.kinds_difficulty.append(1)

        self.kinds_name.append("ランダム")
        self.kinds_func.append([self.human.com_random])
        self.kinds_difficulty.append(1)

        self.kinds_name.append("ランダム-チート")
        self.kinds_func.append([self.human.com_cheater1])
        self.kinds_difficulty.append(1)

        self.kinds_name.append("MC木探索")
        self.mcts_d0 = MonteCarloTreeSearch(1024*1)
        self.mcts_d1 = MonteCarloTreeSearch(1024*4)
        self.mcts_d2 = MonteCarloTreeSearch(1024*16)
        self.mcts_d3 = MonteCarloTreeSearch(1024*64)
        self.kinds_func.append([ self.mcts_d0, self.mcts_d1, self.mcts_d2, self.mcts_d3])
        self.kinds_difficulty.append(4)

        self.kinds_name.append("原始MC探索")
        self.kinds_func.append([PrimitiveMonteCarlo(256*1), PrimitiveMonteCarlo(256*4), PrimitiveMonteCarlo(256*16), PrimitiveMonteCarlo(256*32)])
        self.kinds_difficulty.append(4)

        self.kinds_name.append("原始MC探索 + NegaAlpha")
        self.kinds_func.append([NAPrimitiveMonteCarlo(256*1, 2), NAPrimitiveMonteCarlo(256*4, 4), NAPrimitiveMonteCarlo(256*16, 8), NAPrimitiveMonteCarlo(256*32, 16)])
        self.kinds_difficulty.append(4)

        self.kinds_name.append("AlphaBeta")
        self.kinds_func.append([AlphaBeta(0), AlphaBeta(1)])
        self.kinds_difficulty.append(2)

        self.rainbow_computer_d0 = RainbowComputer(64)
        self.kinds_name.append("Rainbow")
        self.kinds_func.append([ self.rainbow_computer_d0 ])
        self.kinds_difficulty.append(1)

        self.reinforce_computer_d0 = ReinforceComputer(64)
        self.kinds_name.append("Reinforce")
        self.kinds_func.append([ self.reinforce_computer_d0 ])
        self.kinds_difficulty.append(1)

        self.alphazero_computer_d0 = AlphaZeroComputer(64)
        self.kinds_name.append("AlphaZero")
        self.kinds_func.append([ self.alphazero_computer_d0 ])
        self.kinds_difficulty.append(1)

    def get_num(self):
        return len(self.kinds_name)

    def get_name(self, id):
        if id<0 or id>=len(self.kinds_name):
            print("範囲外のIDが指定されました")
            exit()
        return self.kinds_name[id]

    def get_func(self, id, diff):
        if id<0 or id>=len(self.kinds_func):
            print("範囲外のIDが指定されました")
            exit()
        if diff<0 or diff>=len(self.kinds_difficulty):
            print("範囲外のIDが指定されました")
            exit()
        return self.kinds_func[id][diff]

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
        self.width = 640
        self.height = 480
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
    # gameの設定
    def game_config(self, player1id, player2id, player1diff=0, player2diff=0):
        self.player_kinds.alphazero_computer_d0.reset("alphazero-6")
        self.player_kinds.mcts_d0.reset()
        self.player_kinds.mcts_d1.reset()
        self.player_kinds.mcts_d2.reset()
        self.player_kinds.mcts_d3.reset()
        player1_plan = self.player_kinds.get_func(player1id, player1diff)
        player2_plan = self.player_kinds.get_func(player2id, player2diff)
        self.set_plan(player1_plan, player2_plan)

    def render(self, mask, flag, n = 999):
        self.add_playlog()
        self.reversed = mask
        self.main_window.game_page.canvas_update(flag, n)


    def play(self):
        # ウインドウ
        self.main_window = MainWindow(self)

        # プレイヤーの種類
        self.player_kinds = PlayerKinds(self.main_window)

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
    board = Board()
    print("サーバーのIPを入力してください-ない場合は0に")
    ip = input()
    IPADDR = ip
    displayboard = DisplayBoard()
    displayboard.play()
    exit()