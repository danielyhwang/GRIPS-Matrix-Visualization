import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6 import QtCore

from designer_test.ui_mainwindow import Ui_MainWindow
# If you are getting Ui_MainWindow not defined, try running pyside6-project build
# and then run pyside6-project run
# https://stackoverflow.com/questions/62697494/python-qt-no-module-named-ui-mainwindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)
        
        # testing adding custom functionality. test kinda failed, you need to copy in the set from ui_mainwindow.py first into this window.
        #self.uploadButton.clicked.connect(self.changeText)

    #@QtCore.Slot()
    #def changeText(self):
        #self.text.setText("Changed!")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    #https://blog.csdn.net/m0_56960619/article/details/125239865
    #commented out, alternative approach.
    #ui = Ui_Dialog()
    #ui.setupUi(window)
    window.show()
    sys.exit(app.exec())
