## Import

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2
import time
import os.path
import tensorflow as tf 
import keras
from keras import layers
from ComputerControl import *
import sys
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QApplication
import pickle
import gui
MAX_IMAGES_SAVED = 10
currentId = 0



def recognise_hand(images):
    images = images.reshape(1,37, 100, 176, 3)
    model = tf.keras.models.load_model("Test_save_model")
    print(np.argmax(model.predict(images)[0]))
    return np.argmax(model.predict(images)[0])
    
## Helper functions

def get_pictures(sleep_time, window):
    # define a video capture object
    vid = cv2.VideoCapture(0)
    # define the input size of the model
    images_count = 37
    input_shape = (176, 100)
    # define the images list
    width = input_shape[0]
    height = input_shape[1]
    images_shape = (images_count, height, width, 3)
    images = np.empty(images_shape, dtype='uint8')

    for i in range(images_count):
        # Capture the video frame by frame
        ret, frame = vid.read()
        # Resize the frame
        time.sleep(sleep_time)
        img = Image.fromarray(frame)
        resized_img = img.resize(input_shape, Image.LANCZOS)
        frame_numpy = np.array(resized_img)
        
        # Add it to the array
        images[i] = cv2.cvtColor(frame_numpy, cv2.COLOR_BGR2RGB)
        window.update_image(images[i])
        
    
    # After the loop release the cap object
    vid.release()

    return images

def display_images(images): ### For test
    for i in range(images.shape[0]):
        plt.imshow(images[i])
        plt.show()



def calc_fps():
    video = cv2.VideoCapture(0);
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
    num_frames = 120
    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    start = time.time()
 
    for i in range(0, num_frames) :
        ret, frame = video.read()
    
    end = time.time()

    seconds = end - start
    return num_frames / seconds
def get_fps():
    if(os.path.isfile("maxFps.bin")):
        f = open("maxFps.bin", 'rb')
        fps = float(f.read().decode())
        f.close()
        return fps
    fps = calc_fps()
    f= open("maxFps.bin", 'wb')
    f.write(str(fps).encode())
    return fps

def calc_sleep(fps, target_fps):
    if (target_fps>fps):
        raise Exception('Target fps is higher then max fps') 
    diff = fps-target_fps
    return diff/37



## Get the images to the  model 

def input_to_model(sleep_time, window):
    images = get_pictures(sleep_time, window)
   # display_images(images) # test
    id = recognise_hand(images)
    save_images(images, id)
    return id
    


def save_images(images, predictedID):
    f = open("PredictionImages_"+currentId, "wb")
    f.write(pickle.dump([images, predictedID]))
    f.close()    
    currentId+=1
    if currentId>MAX_IMAGES_SAVED:
        currentId = 0

def load_images():
    data = []
    for i in range(MAX_IMAGES_SAVED):
        try:
            f = open("PredictionImages_"+currentId, "rb")
            data += pickle.load(f.read())
        except:
            return data
    return data



"""fps = get_fps()
print("Max fps is: " +str(fps))
target_fps = float(input("Input target fps:"))
app = QApplication(sys.argv)
window = QMainWindow()
window.show()

while True:
    
    id = input_to_model(calc_sleep(fps, target_fps),window)
    execute_command(id_to_command(id))"""
