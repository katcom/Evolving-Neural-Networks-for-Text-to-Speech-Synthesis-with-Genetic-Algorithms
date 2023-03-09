import socket

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

client.connect(("127.0.0.1",10888))
while True:
    inp = input("S  end to server:")
    client.send(inp.encode())
    print("Message from server:")
    print(client.recv(4096))