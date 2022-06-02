import socket
import threading
import datetime
from netifaces import interfaces, ifaddresses, AF_INET
import time
import sys
#from board import board

IPADDR = "127.0.0.1"
PORT = 10100


class Client:
    def __init__(self):
        self.client_id = -1
        self.sock = None
        self.recv_func = None
        self.recv_thread = None
        pass
    
    # 通信を開始する
    def connect(self, ip="127.0.0.1", port=8080):
        self.client_id = 0
        self.sock = socket.socket(socket.AF_INET)
        self.sock.connect((ip, port))
        pass

    # サーバーからデータを受信する
    def recv_loop(self):
        while True:
            try:
                data = self.sock.recv(1024)
                if data == b"":
                    break
                print("受信データ:", data.decode("utf-8"))
                self.__recv_func(  data.decode("utf-8")  )
                self.recv_func(  data.decode("utf-8") )
            except ConnectionResetError:
                break
            #except:
                break
        print("通信を終了します")
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()
    
    def __recv_func(self, x):
        y = x.split()
        if len(y)==0:return
        if y[0]=="SET":
            if len(y)<2:
                return
            if y[1]=="ID":
                if len(y)<3:
                    return
                self.client_id = int(y[2])
        pass

    def send(self, message):
        if self.sock is None:
            return
        try:
            self.sock.send(message.encode("utf-8"))
        except:
            print("送信エラーが発生しました：クライアント")
        pass



class NetrorkPlayer():
    def __init__(self, ip):
        self.NoNetwork = (ip=="0")
        if self.NoNetwork :
            return

        self.client = Client()
        self.client.connect(str(ip), PORT)
        self.client.recv_func = self.recv_func
        self.thread = threading.Thread(target=self.client.recv_loop, args=())
        self.thread.daemon = True
        self.thread.start()

        self.NoNetwork = True
        self.put_place = -1

    def reset(self):
        pass
    
    def next_action(self, board):
        if self.NoNetwork:
            return 0
        placable = set(board.list_placable())
        print(placable)
        while self.put_place<0 or (not (self.put_place in placable)):
            time.sleep(0.1)
        ret = self.put_place
        self.put_place  = -1
        return ret
    
    def notice(self, p):
        if self.NoNetwork:
            return
        s = "PUT " + str(p)
        self.client.send(s)

    def recv_func(self, data):
        if self.NoNetwork:
            return
        y = data.split()
        if len(y)<2:
            return
        if y[0]=="PUT":
            self.put_place =  int(y[1])
        pass
    








