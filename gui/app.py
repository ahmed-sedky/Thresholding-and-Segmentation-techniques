from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow
import sys
import modules.utils as Utils


class Ui(QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi("app.ui", self)
        self.template_widget.setVisible(False)
        self.showMaximized()
        self.open_image.clicked.connect(lambda: Utils.browse_files(self, "original"))
        self.open_template_image.clicked.connect(
            lambda: Utils.browse_files(self, "template")
        )
        self.features_combobox.activated[str].connect(
            lambda text: Utils.choose_feature(self, text)
        )


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    window.show()
    sys.exit(app.exec_())
