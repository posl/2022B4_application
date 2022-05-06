import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import random
import board

#from turtle import width

#ウィンドウの高さと幅を設定
window_width = 640
window_height = 480
canvas_width = 400
canvas_height = 400
cell_width = 48
cell_height = 48
font_name = "ヒラギノ"




#windowを使えるようにする
tkapp = tk.Tk()
tkapp.title("おせろー ver 1.00")
tkapp.geometry(str(window_width) + "x" + str(window_height))
tkapp.grid_columnconfigure(0, weight=1)
tkapp.grid_rowconfigure(0, weight=1)

#ゲーム時に表示されるフレーム
start_frame = tk.Frame(tkapp, bg = "#FF9999")
start_frame.grid(row=0, column=0, sticky="nsew")

start_label = tk.Label(start_frame, text="Othello", bg="#FF9999", fg="#99FF99", font = (font_name,50))
#start_label.place(x=200, y=30 )
start_label.pack(anchor="center", expand=True)

gamestart_button = tk.Button(start_frame, text="START", font=(font_name,50), width=6, height=1, command=lambda:option_frame.tkraise())
#gamestart_button.place(x=200, y=250)
gamestart_button.pack(anchor="center", expand=True)

quit_button = tk.Button(start_frame, text="QUIT", font=(font_name,50), width=6, height=1, command=lambda:tkapp.quit())
#quit_button.place(x=200, y=350)
quit_button.pack(anchor="center", expand=True)



#オプション選択画面
option_frame = tk.Frame(tkapp, bg = "#FF9999")
option_frame.grid(row=0, column=0, sticky="nsew")

player1_label = tk.Label(option_frame, text="先攻", bg="#FF9999", fg="#111111", font = (font_name,20))
player1_label.place(x=50, y=30 )

player2_label = tk.Label(option_frame, text="後攻", bg="#FF9999", fg="#111111", font = (font_name,20))
player2_label.place(x=50, y=90 )


combobox_menus = ('手動', 'COM')

player1_combobox = ttk.Combobox(option_frame, height=3, values = combobox_menus, state="readonly")
player1_combobox.place(x=150, y=30 )
player1_combobox.current(0)

player2_combobox = ttk.Combobox(option_frame, height=3, values = combobox_menus, state="readonly")
player2_combobox.place(x=150, y=90 )
player2_combobox.current(1)


def player(board : board.Board):
	while 1:
		try:
			n = int(input("enter n : "))
			if board.is_placable(n):
				return n
		except:
			print("error")
			continue

def com_random(board : board.Board):
	return random.choice(board.list_placable())

def next_button1_click():
	game_frame.tkraise()
	game_board.set_plan(com_random, com_random, 1)

next_button1 = tk.Button(option_frame, text="NEXT", font=(font_name,50), width=6, height=1, command=lambda:next_button1_click())
next_button1.place(x=400, y=400 )




#
game_frame = tk.Frame(tkapp, bg = "#44EE88")
game_frame.grid(row=0, column=0, sticky="nsew")

game_canvas = tk.Canvas(game_frame, width=400, height=400)
game_canvas.place(x=50, y=30)

def game_button1_click():
	game_board.game()
	game_canvas_update()

game_button1 = tk.Button(game_frame, text="AUTO", font=(font_name,50), width=6, height=1, command=lambda:game_button1_click())
game_button1.place(x=500, y=30)


# ボード
game_board = board.Board()
#game_board.stone_exist = 0xffffffffff000fff
#game_board.stone_black = 0x000f0f0f

# 盤面を描画する関数
def game_canvas_update():
	game_canvas.configure(bg="#44EE88")
	game_canvas.create_rectangle(0, 0, canvas_width+10, canvas_height+10, fill = "#22FF77")
	for i in range(9):
		game_canvas.create_line(10+cell_width*i, 10, 10+cell_width*i, 10+cell_height*8, fill="#101010", width=2)
		game_canvas.create_line(10, 10+cell_height*i, 10+cell_width*8, 10+cell_height*i, fill="#101010", width=2)
	for i in range(8):
		for j in range(8):
			t = (j, i)
			if (game_board.stone_exist >> (board.Board.t2n(t)) &1)==1 :
				if (game_board.stone_black >> (board.Board.t2n(t)) &1)==1 :
					game_canvas.create_oval(11+cell_width*i, 11+cell_height*j, 9+cell_width*(i+1), 9+cell_height*(j+1), fill="#111111")
				else:
					game_canvas.create_oval(11+cell_width*i, 11+cell_height*j, 9+cell_width*(i+1), 9+cell_height*(j+1), fill="#EEEEEE")





game_canvas_update()

#ゲーム開始のフレームを最前面に持ってくる
game_frame.tkraise()
start_frame.tkraise()



def fix():
    a = tkapp.winfo_geometry().split('+')[0]
    b = a.split('x')
    w = int(b[0])
    h = int(b[1])
    tkapp.geometry('%dx%d' % (w+1,h+1))

tkapp.update()

tkapp.after(0, fix)

#無限ループで処理を行う
tkapp.mainloop()

