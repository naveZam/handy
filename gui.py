import sys
import PyQt5
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QApplication
from PyQt5.QtGui import QPixmap, QImage
from camera import *
class ImageWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.image_label = QLabel()
        self.image_label.setScaledContents(True)

        self.central_widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.central_widget.setLayout(layout)

        self.setCentralWidget(self.central_widget)
        self.setWindowTitle("Image Viewer")

        # Connect the resize event to update the image
        self.resizeEvent = self.update_image_size
    def update_image(self, image_array):
        """Updates the displayed image using a NumPy array.

        Args:
            image_array: A NumPy array representing the image data.
        """

        # Convert NumPy array to QImage
        qimage = QImage(image_array.data, image_array.shape[1], image_array.shape[0], 
                        image_array.strides[0])

        # Convert QImage to QPixmap and set on the label
        pixmap = QPixmap(qimage).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.SmoothScaling)
        self.image_label.setPixmap(pixmap)

    def update_image_size(self, event):
        """Updates the image size based on the window size."""
        if self.image_label.pixmap():
            self.update_image(self.image_label.pixmap().toImage().copy())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageWindow()
    window.show()
    while True:
        id = input_to_model(calc_sleep(get_fps(), get_fps()-1),window.update_image)
        print(1)  
        sys.exit(app.exec_())
    