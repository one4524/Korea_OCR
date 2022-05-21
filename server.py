import socket
import time
from main_ocr import main

host = 'localhost'  # Symbolic name meaning all available interfaces
port = 7070  # Arbitrary non-privileged port

server_sock = socket.socket(socket.AF_INET)
server_sock.bind((host, port))
server_sock.listen(1)


print("기다리는 중")


def get_bytes_stream(sock, length):
    buffer = b''
    try:
        remain = length
        while True:
            data = sock.recv(remain)
            buffer += data
            if len(buffer) == length:
                break
            elif len(buffer) < length:
                remain = length - len(buffer)
    except Exception as e:
        print(e)
    return buffer[:length]


def write_utf8(data, sock):
    encoded = data.encode(encoding='utf-8')
    sock.sendall(len(encoded).to_bytes(4, byteorder="big"))
    sock.sendall(encoded)


while True:
    client_sock, addr = server_sock.accept()

    print('Connected by', addr)

    len_bytes_string = bytearray(client_sock.recv(1024))[2:]
    len_bytes = len_bytes_string.decode("utf-8")
    length = int(len_bytes)

    img_bytes = get_bytes_stream(client_sock, length)

    with open("res_image.jpg", "wb") as writer:
        writer.write(img_bytes)

    print("img is saved")

    price, date = main("res_image.jpg")

    data = "admin,매장,"+str(price)+date[0:5]+","+date[5:7]+","+date[7:9]
    write_utf8(data, client_sock)

