import socket
import time


host = ''
port = 10888

udp_server = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
udp_server.bind((host,port))
#udp_server.listen(0)

print("start TTS server")
while True:
    data,address = udp_server.recvfrom(4096)
    print("get connection from",address)
    msg = "Hello from server. Echo:"+data.decode()
    udp_server.sendto(msg.encode(),address)

udp_server.close()

