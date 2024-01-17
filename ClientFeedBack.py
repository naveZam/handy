import socket
import os
import zipfile
import pickle
HOST = "127.0.0.1"
PORT = 65432  
MODEL_NAME = "model"
def sendFeedBack(images, correctID):

    data = [images, correctID]
    data = pickle.dumps(data)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.sendall(data)
    s.close()


def requestUpdate():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.sendall("requestUpdate".encode())
    zip_size = int(s.recv(1024))
    f = open(MODEL_NAME, "wb")
    f.write(s.recv(zip_size))
    f.close()
    s.close()
    with zipfile.ZipFile(MODEL_NAME, "r") as z:
        z.extractall()
    os.remove(MODEL_NAME)


