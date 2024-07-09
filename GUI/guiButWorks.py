import sys
import cv2
import os
from PyQt5.QtCore import QTimer
import functools
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from ComputerControl import id_to_command
import numpy as np
from PIL import Image
import camera
import threading
IMAGES_DIRECTORY = "saved_images"
MAX_IMAGES = 10
IMAGES_COUNT = 37

class VideoDisplay(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Webcam Video Display")
        self.video_label = QLabel()
        self.number_input = QLineEdit()
        self.execute_button = QPushButton("Execute")
        
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.number_input)
        layout.addWidget(self.execute_button)
        self.setLayout(layout)
        vid = cv2.VideoCapture(0)
        # define the input size of the model
        self.id = 0
        input_shape = (176, 100)
        # define the images list
        width = input_shape[0]
        height = input_shape[1]
        images_shape = (IMAGES_COUNT, height, width, 3)
        images = np.empty(images_shape, dtype='uint8')
        self.video_capture = cv2.VideoCapture(0)

        timerCallback = functools.partial(self.display_video, images, input_shape)

        self.timer = QTimer()
        self.timer.timeout.connect(timerCallback)
        self.timer.start(73)  # Update video every 30 milliseconds

        
        
        self.execute_button.clicked.connect(self.execute_command)
        self.image_counter = 0

        if not os.path.exists(IMAGES_DIRECTORY):
            os.makedirs(IMAGES_DIRECTORY)

    def display_video(self, images, input_shape):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap)
            self.resize(pixmap.width(), pixmap.height())


        img = Image.fromarray(frame)
        resized_img = img.resize(input_shape, Image.LANCZOS)
        frame_numpy = np.array(resized_img)
        
        # Add it to the array
        images[self.id] = cv2.cvtColor(frame_numpy, cv2.COLOR_BGR2RGB)
        self.id= self.id+1
        if self.id > IMAGES_COUNT-1:
            self.id = 0
            thread = threading.Thread(target=camera.recognise_hand,args=[images])
            thread.start()
    
    def execute_command(self):
        try:
            number = int(self.number_input.text())
            id_to_command(number)
        except:
            return


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoDisplay()
    window.show()
    sys.exit(app.exec_())