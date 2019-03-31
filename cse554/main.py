from PyQt5.QtWidgets import QApplication
from gui import GUI
import sys
import shutil
import os

if __name__ == '__main__':

    if os.path.exists('./temp/'):
        shutil.rmtree('./temp/', ignore_errors=True)
    os.makedirs('./temp/')

    app = QApplication(sys.argv)
    gui = GUI()
    sys.exit(app.exec_())
