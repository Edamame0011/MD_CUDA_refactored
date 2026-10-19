"""Start with either python ui/main.py or python -m ui.main."""
import sys
from PySide6.QtWidgets import QApplication

if __package__:
    from .core.main_window import MainWindow
else:
    from core.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.deleteLater)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
