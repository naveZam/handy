import socket
import os
import zipfile
import shutil
import pickle
import threading

feedbacks = []
MODEL_NAME = "model"
def sendModel(c):
    shutil.make_archive(MODEL_NAME, "zip", MODEL_NAME)
    size = os.path.getsize(MODEL_NAME+".zip")
    c.send(str(size).encode())
    file = open(MODEL_NAME+".zip", "rb")
    c.send(file.read())
    c.close()

def receiveFeedBack(c):
    message = c.recv(1024)
    feedbacks.append(pickle.load(message))
    with open('feedbackData.pickle', 'wb') as handle:
        pickle.dump(feedbacks, handle, protocol=pickle.HIGHEST_PROTOCOL)

def trainModel():
    if len(feedbacks) > 10:
        print("training model")

     


with open('feedbackData.pickle', 'rb') as handle:
    feedbacks = pickle.loads(handle.read())
s = socket.socket()
trainingThread = threading.Thread(trainModel)
trainingThread.start()
while True:
    s.bind(("localhost", 5542))
    s.listen(1)
    c, addr = s.accept()
    message = s.recv(1024)
    if message == "requestUpdate":
        threading.Thread(sendModel(c)).start()
    else:
        threading.Thread(receiveFeedBack(c)).start()
    

