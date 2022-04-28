import tkinter as tk
master = tk.Tk()
w = tk.Canvas(master, width=640, height=480)
w.pack()

cellwidth = 40
cellheight = 40

w.create_rectangle(0, 0, 640, 480, fill="#20B040")


a = []
for i in range(8):
    a.append([])

for i in range(8):
    for j in range(8):
        a[i].append(0)

print(a)

def paint(event):
    x = event.x
    y = event.y
    x = (x - 100) // cellwidth
    y = (y - 100) // cellheight
    if x>=0 and x < 8 and y>=0 and y<8:
        a[x][y] = (a[x][y]+1) % 3
    w.create_rectangle(0, 0, 640, 480, fill="#20B040")
    for i in range(9):
        w.create_line(100+cellwidth*i, 100, 100+cellwidth*i, 100+cellheight*8, fill="#101010", width=2)
        w.create_line(100, 100+cellheight*i, 100+cellwidth*8, 100+cellheight*i, fill="#101010", width=2)
    
    for i in range(8):
        for j in range(8):
            if a[i][j] == 0 :
                w.create_oval(102+cellwidth*i, 102+cellheight*j, 98+cellwidth*(i+1), 98+cellheight*(j+1), fill="#111111")
            if a[i][j] == 1 :
                w.create_oval(102+cellwidth*i, 102+cellheight*j, 98+cellwidth*(i+1), 98+cellheight*(j+1), fill="#EEEEEE")
            




w.create_rectangle(50, 20, 150, 80, fill="#476042")
w.create_rectangle(65, 35, 135, 65, fill="yellow")
w.create_line(0, 0, 50, 20, fill="#476042", width=3)
w.create_line(0, 100, 50, 80, fill="#476042", width=3)
w.create_line(150,20, 200, 0, fill="#476042", width=3)
w.create_line(150, 80, 200, 100, fill="#476042", width=3)
w.bind("<Button-1>", paint)
master.mainloop()