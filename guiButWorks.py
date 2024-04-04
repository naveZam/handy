import sys
import cv2
import os
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton
from ComputerControl import id_to_command

IMAGES_DIRECTORY = "saved_images"
MAX_IMAGES = 10

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

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video)
        self.timer.start(30)  # Update video every 30 milliseconds

        self.video_capture = cv2.VideoCapture(0)
        
        self.execute_button.clicked.connect(self.execute_command)
        self.image_counter = 0

        if not os.path.exists(IMAGES_DIRECTORY):
            os.makedirs(IMAGES_DIRECTORY)

    def display_video(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap)
            self.resize(pixmap.width(), pixmap.height())
    
    def execute_command(self):
        number = int(self.number_input.text())
        id_to_command(number)

        # Save the image
        if self.image_counter < MAX_IMAGES:
            file_name = f"{IMAGES_DIRECTORY}/image_{self.image_counter}.jpg"
        else:
            oldest_file = f"{IMAGES_DIRECTORY}/image_{self.image_counter % MAX_IMAGES}.jpg"
            os.remove(oldest_file)
            file_name = f"{IMAGES_DIRECTORY}/image_{self.image_counter % MAX_IMAGES}.jpg"

        ret, frame = self.video_capture.read()
        if ret:
            cv2.imwrite(file_name, frame)
            self.image_counter += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoDisplay()
    window.show()
    sys.exit(app.exec_())