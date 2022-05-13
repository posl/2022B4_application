import tkinter as tk
import tkinter.ttk as ttk
import math



class Page(tk.Frame):
    def __init__(self):
        # ページに共通する属性やメソッドなどを記述
        self.win_width = 640
        self.win_height = 480
        self.grid(row=0, column=0, sticky="nsew") #正常な画面表示に必要
        



class StartPage(Page):
    def __init__(self, board):
        tk.Frame.__init__(self, board)
        self.board = board
        self.configure(bg="#881111")

        self.label = tk.Label(self, text="Othello", font = (board.font_name, 50), fg="#119911", bg="#881111")
        self.label.pack(anchor="center", expand=True)
        #self.label.place(x=200, y=50)
        self.button1 = tk.Button(self, text="Play", font = (board.font_name, 50), command=lambda:self.goto_option_page())
        self.button1.pack(anchor="center", expand=True)
        #self.button1.place(x=200, y=200)
        self.button2 = tk.Button(self, text="Quit", font = (board.font_name, 50), command=lambda:board.quit())
        self.button2.pack(anchor="center", expand=True)
        #self.button2.place(x=200, y=270)

    def goto_option_page(self):
        self.board.option_page.tkraise()
        self.board.quit()


class OptionPage(Page):
    def __init__(self, board):
        tk.Frame.__init__(self, board)
        self.board = board
        self.configure(bg="#992299")
        self.grid(row=0, column=0, sticky="nsew")

        self.combo_menus = ("手動", "COM-A(ランダム)", "COM-B(モンテカルロ)", "COM-C(alpha)")

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

        self.button1 = tk.Button(self, text="Next", font = (board.font_name, 50), command=lambda:self.start_game())
        self.button1.place(x=450, y=380)

    def start_game(self):
        self.board.reset()
        self.board.game_playing = 1
        if False:
            self.board.stone_black = 0
            for i in range(7):
                for j in range(7):
                    self.board.stone_exist = self.board.stone_exist | (1<<(8*i+j))
                    if (i+j)%2==0:
                        self.board.stone_black = self.board.stone_black + (1<<(8*i+j))
                        pass
                    else:
                        pass
        self.board.game_page.player1 = self.combobox1.current()
        self.board.game_page.player2 = self.combobox2.current()
        player1_plan = 0
        if self.combobox1.current() == 0:
            print("先攻：",0)
            player1_plan = self.board.com_random
        elif self.combobox1.current() == 1:
            print("先攻：",1)
            player1_plan = self.board.com_random
        elif self.combobox1.current() == 2:
            print("先攻：",2)
            player1_plan = self.board.com_monte
        elif self.combobox1.current() == 3:
            print("先攻：",3)
            player1_plan = self.board.com_alpha0
        if self.combobox2.current() == 0:
            print("後攻",0)
            player2_plan = self.board.com_random
        elif self.combobox2.current() == 1:
            print("後攻",1)
            player2_plan = self.board.com_random
        elif self.combobox2.current() == 2:
            print("後攻",2)
            player2_plan = self.board.com_monte
        elif self.combobox2.current() == 3:
            print("後攻",3)
            player2_plan = self.board.com_alpha1
        self.board.set_plan(player1_plan, player2_plan)
        self.board.game_page.canvas_update()
        self.board.game_page.tkraise()
        self.board.after(1000, self.board.game_page.button1_click)




class GamePage(Page):
    def __init__(self, board):
        tk.Frame.__init__(self, board)
        self.board = board
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

        self.black_conter_label = tk.Label(self, text="B00", fg="#111111", bg="#808080", font = (board.font_name, 50))
        self.black_conter_label.place(x=10, y=10)

        self.white_conter_label = tk.Label(self, text="W00", fg="#EEEEEE", bg="#808080", font = (board.font_name, 50))
        self.white_conter_label.place(x=530, y=10)

        self.label1 = tk.Label(self, text="", fg="#111111", bg="#808080", font = (board.font_name, 25))
        self.label1.place(x=1000, y=300)

        self.button1 = tk.Button(self, text="Next", font = (board.font_name, 25), command=lambda:self.button1_click())
        self.button1.place(x=750, y=380)

        self.button2 = tk.Button(self, text=">", font = (board.font_name, 50), command=lambda:self.goto_start_page())
        self.button2.place(x=750, y=380)
    
    def canvas_update(self, state=0, x=0, y=0, oldcolor=0):
        if state==0:
            self.board.se1.play()
        elif state==1:
            self.board.se2.play()
        elif state==2:
            self.board.se3.play()
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
                if (self.board.stone_exist >> (self.board.t2n(t)) &1)==1 :
                    if (self.board.stone_black >> (self.board.t2n(t)) &1)==1 :
                        self.stone_black_draw(i, j)
                    else:
                        self.stone_white_draw(i, j)
        lp = self.board.list_placable()
        if state==0:
            for w in lp:
                i = w%8
                j = w//8
                self.stone_blue_draw(i,j)
        if state==1:
            self.stone_yellow_draw(x,y)
            self.board.after(800, self.canvas_update, 2, x, y, self.board.stone_black)
        if state==2:
            for i in range(8):
                for j in range(8):
                    t = (j, i)
                    if (self.board.stone_exist >> (self.board.t2n(t)) &1)==1 :
                        if (self.board.stone_black >> (self.board.t2n(t)) &1) ^ (oldcolor >> (self.board.t2n(t)) &1)==1 :
                            self.stone_gray_draw(i, j)
            self.stone_yellow_draw(x,y)
            self.board.after(800, self.canvas_update, 0, 0, 0, 0)
            self.game_exit_check()

    def render_current_board(self):
        bnum = 0
        wnum = 0
        self.game_canvas.delete("all")
        self.game_canvas.configure(bg="#44EE88")
        self.game_canvas.create_rectangle(0, 0, self.canvas_width+10, self.canvas_height+10, fill = "#22FF77")
        for i in range(9):
            self.game_canvas.create_line(10+self.cell_width*i, 10, 10+self.cell_width*i, 10+self.cell_height*8, fill="#101010", width=2)
            self.game_canvas.create_line(10, 10+self.cell_height*i, 10+self.cell_width*8, 10+self.cell_height*i, fill="#101010", width=2)
        for i in range(8):
            for j in range(8):
                t = (j, i)
                if (self.board.stone_exist >> (self.board.t2n(t)) &1)==1 :
                    if (self.board.stone_black >> (self.board.t2n(t)) &1)==1 :
                        self.stone_black_draw(i, j)
                    else:
                        self.stone_white_draw(i, j)
        return
    
    def render_placeable(self):
        lp = self.board.list_placable()
        for w in lp:
            i = w%8
            j = w//8
            self.stone_blue_draw(i,j)
        return

    def render_reverse(self, n, flg = True):
        y, x = self.board.n2t(n)
        self.stone_yellow_draw(x, y)
        r_list = self.board.__reverse(n)
        if flg:
            for i in r_list:
                i_y, i_x = self.board.n2t(i)
                self.stone_gray_draw(i_x, i_y)
            self.board.after(800, self.render_reverse, False)
        else:
            for i in r_list:
                i_y, i_x = self.board.n2t(i)
                self.stone_gray_draw(i_x, i_y)
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
                if (self.board.stone_exist >> (self.board.t2n(t)) &1)==1 :
                    if (self.board.stone_black >> (self.board.t2n(t)) &1)==1 :
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
        if self.board.turn==1 and self.player1==0:
            print("COMは石を置けなかった(先攻)")
            return
        if self.board.turn==1 and self.player1==1:
            print()
        if self.board.turn==1 and self.player1==2:
            print()
        if self.board.turn==1 and self.player1==3:
            print()
        if self.board.turn==0 and self.player2==0:
            print("COMは石を置けなかった(後攻)")
            return
        if self.board.turn==0 and self.player2==1:
            print()
        if self.board.turn==0 and self.player2==2:
            print()
        if self.board.turn==0 and self.player2==3:
            print()
        self.game_still_cont = self.board.can_continue(True)
        self.game_still_cont = self.board.can_continue(True)
        if self.game_still_cont:
            n = self.board.get_action()
            self.canvas_update(1, n%8, n//8)
            self.board.put_stone(n)
            self.game_still_cont = self.board.can_continue()
        print("COMが石を置いた")
        if self.board.game_playing:
            self.board.after(1800, self.button1_click)
    
    def human_put_stone(self, x, y):
        if x<0 or y<0 or x>=8 or y>=8:
            self.board.se4.play()
            return
        t = (y, x)
        if (self.board.stone_exist >> self.board.t2n(t)) & 1:
            print("既に石があります")
            self.board.se4.play()
            return
        if self.board.is_placable(self.board.t2n(t))==False:
            print("石を底に置くことはできません")
            self.board.se4.play()
            return
        self.canvas_update(1, x, y)
        self.game_still_cont = self.board.can_continue(True)
        self.game_still_cont = self.board.can_continue(True)
        if self.game_still_cont:
            n = self.board.t2n(t)
            self.board.put_stone(n)
            self.game_still_cont = self.board.can_continue()
        if self.board.game_playing:
            self.board.after(1800, self.button1_click)
        return

    def game_exit_check(self):
        flg = 0
        flg = self.board.can_continue(True)
        flg = self.board.can_continue(True)
        if flg:
            self.board.game_playing = 1
            return
        self.board.game_playing = 0
        self.board.after(1500, self.goto_result_page)

    def cell_click(self, event):
        #print(self.board.turn)
        #print(self.player1)

        if self.board.turn==1 and self.player1==0:
            print()
        if self.board.turn==1 and self.player1>0:
            self.board.se4.play()
            print("あなたの番ではありません：")
            return
        if self.board.turn==0 and self.player2==0:
            print()
        if self.board.turn==0 and self.player2>0:
            self.board.se4.play()
            print("あなたの番ではありません：")
            return
        x = event.x
        y = event.y
        x = (x-10) // self.cell_width
        y = (y-10) // self.cell_height
        if x<0 or y<0 or x>=8 or y>=8:
            self.board.click_flag = -1
        self.human_put_stone(x, y)
        #self.canvas_update()
        print("セルが押された：", x, ",", y)

        #self.board.player_action = n


    def goto_result_page(self):
        self.win_check()
        self.board.result_page.win_check()
        self.button2.place(x=520)
        self.label1.place(x=520)

    def goto_start_page(self):
        self.board.start_page.tkraise()
        self.button2.place(x=1000)
        self.label1.place(x=1000)
        self.label1.configure(text="")
    

    def win_check(self):
        bnum = 0
        wnum = 0
        for i in range(8):
            for j in range(8):
                t = (j, i)
                if (self.board.stone_exist >> (self.board.t2n(t)) &1)==1 :
                    if (self.board.stone_black >> (self.board.t2n(t)) &1)==1 :
                        bnum += 1
                    else:
                        wnum += 1
        if bnum>wnum:
            self.label1.configure(text="黒の勝ち")
        elif bnum<wnum:
            self.label1.configure(text="白の勝ち")
        else:
            self.label1.configure(text="引き分け")







if __name__ == "__main__":

    # アプリケーションを開始
    #board = board.Board()
    #board.play()
    
    #tkapp = App()
    #tkapp.title('おせろ')
    
    #tkapp.mainloop()
    #board.set_display(tkapp)

    def player(board):
        placable = set(board.list_placable())
        while True:
            board.main_loop()
            n = board.click_flag
            if n in placable:
                break
        return n