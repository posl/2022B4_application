import socket
import threading
import time

from netifaces import interfaces, ifaddresses, AF_INET

PORT = 10100



class Client:
    def __init__(self):
        self.client_id = -1
        self.sock = None
        self.recv_func = None
        self.recv_thread = None

    # 通信を開始する
    def connect(self, ip="127.0.0.1", port=8080):
        self.client_id = 0
        self.sock = socket.socket(socket.AF_INET)
        self.sock.connect((ip, port))

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

    def send(self, message):
        if self.sock is None:
            return
        try:
            self.sock.send(message.encode("utf-8"))
        except:
            print("送信エラーが発生しました：クライアント")



class NetrorkPlayer():
    ip = None
    client = None
    reset_fin = False
    NoNetwork = False
    thread = None
    put_place = 0

    def __init__(self):
        print(self.reset_fin)
        print("NETWORK PLAYER INIT")
        self.reset()

    def reset(self):
        if NetrorkPlayer.reset_fin==False:
            NetrorkPlayer.reset_fin = True
            ip = NetrorkPlayer.ip
            NetrorkPlayer.NoNetwork = (ip=="0")
            if NetrorkPlayer.NoNetwork :
                return

            NetrorkPlayer.client = Client()
            NetrorkPlayer.client.connect(str(ip), PORT)
            NetrorkPlayer.client.recv_func = NetrorkPlayer.recv_func
            NetrorkPlayer.thread = threading.Thread(target=NetrorkPlayer.client.recv_loop, args=())
            NetrorkPlayer.thread.daemon = True
            NetrorkPlayer.thread.start()

            #self.NoNetwork = True
            NetrorkPlayer.put_place = -1

    def __call__(self, board):
        return self.next_action(board)

    def reset_kesu(self):
        pass

    def next_action(self, board):
        if NetrorkPlayer.NoNetwork:
            return 0
        placable = set(board.list_placable())
        print(placable)
        while NetrorkPlayer.put_place<0 or (not (NetrorkPlayer.put_place in placable)):
            time.sleep(0.1)
        ret = NetrorkPlayer.put_place
        NetrorkPlayer.put_place  = -1
        return ret

    def notice(self, p):
        if NetrorkPlayer.NoNetwork:
            return
        s = "PUT " + str(p)
        print(NetrorkPlayer.client)
        self.client.send(s)

    def recv_func(data):
        if NetrorkPlayer.NoNetwork:
            return
        y = data.split()
        if len(y)<2:
            return
        if y[0]=="PUT":
            NetrorkPlayer.put_place =  int(y[1])




class Server:
    def __init__(self):
        # tupleを格納： id -> (id,IPアドレス,PORT)
        self.client_map = {}
        self.sock = None
        self.client_id_curr = 0
        self.recv_func = None
        self.max_connection = 2

    # 通信を開始
    def connect(self, port="8080"):
        ip = self.get_self_ip_addr()
        self.sock = socket.socket(socket.AF_INET)
        self.sock.bind((ip, port))
        self.sock.listen()

    # 特定のクライアントから受信 スレッドにより実行される
    def recv_client(self, id, sock, addr):
        while True:
            try:
                data = sock.recv(1024)
                if data==b"":
                    break
                print( str(self.client_map[id]) ,"から受信された：", data.decode("utf-8")  )
                self.recv_func( id, data.decode("utf-8") )
            except ConnectionResetError:
                break
            except:
                break
        print("クライアントが退出:", self.client_map[id][0], self.client_map[id][2] )
        del self.client_map[id]

        try:
            sock.shutdown(socket.SHUT_RDWR)
        except:
            pass
        sock.close()

    # 新たなクライアントがいる場合には、スレッドを立てて処理を任せる
    def recv_loop(self):
        while True:
            while len(self.client_map)>=self.max_connection:
                time.sleep(1.0)
            sock_cl, addr = self.sock.accept()
            self.client_map[self.client_id_curr] = (self.client_id_curr, sock_cl, addr )
            id = self.client_id_curr
            self.client_id_curr += 1
            print("新しい人が参加した：", str( self.client_map[id][0] ), str( self.client_map[id][2] ) )

            thread = threading.Thread(target=self.recv_client, args=(id, sock_cl, addr))
            thread.daemon = True
            thread.start()

            self.send(id, "SET ID "+str(id) )

    def send(self, id, message):
        try:
            self.client_map[id][1].send( message.encode("utf-8") )
        except:
            print("送信エラー")

    def get_self_ip_addr(self):
        for ifaceName in interfaces():
            addresses = [i['addr'] for i in ifaddresses(ifaceName).setdefault(AF_INET, [{'addr':'No IP addr'}] )]
            for addr in addresses:
                if addr != 'No IP addr' and addr != "127.0.0.1":
                    print(ifaceName, addr)
                    return addr




class OthelloServer:
    def __init__(self):
        self.server = Server()
        self.server.connect(PORT)
        self.server.recv_func = self.recv_func
        self.thread = threading.Thread(target=self.server.recv_loop, args=())
        self.thread.daemon = True
        self.thread.start()

        self.opponents = {}

    def mainloop(self):
        while True:
            s = input()
            y = s.split()
            if y[0]=="exit":
                break
            elif y[0]=="print":
                if y[1]=="client":
                    print( self.server.client_map )

    def recv_func(self, id, data):
        y = data.split()
        if len(y)<2:
            return
        if y[0]=="PUT":
            place = int(y[1])
            for i in self.server.client_map.keys():
                if i==id:
                    continue
                s = "PUT " + str(place)
                self.server.send(i, s )
                break