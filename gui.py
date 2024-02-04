from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import os 
import sys 
import time 

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'test'
        self.left = 50
        self.top = 50
        self.width = 400
        self.height = 140
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.textbox = QLineEdit(self)
        self.textbox.move(150, 90)
        self.textbox.resize(80,40)
        
        self.show()
        
    def mainLoop():
        
    


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())