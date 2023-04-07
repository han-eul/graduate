import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QIcon

form_secondwindow = uic.loadUiType("secondwindow.ui")[0]

class secondWindow(QDialog,QWidget,form_secondwindow):
    def __init__(self):
        super(secondWindow,self).__init__()
        self.initUI()
        self.show()
        
    def initUI(self):
        self.setupUi(self)
        self.setWindowIcon(QIcon('cloud.png'))
        self.closed.clicked.connect(self.Close)
    
    def Close(self):
        self.close()