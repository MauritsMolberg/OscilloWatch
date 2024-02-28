from src.topsrt.time_window_plot import TimeWindowPlot
from PySide6 import QtWidgets
import sys
import time
import numpy as np
import threading


def main():
    app = QtWidgets.QApplication(sys.argv)

    tw_plot = TimeWindowPlot(n_samples=100, n_cols=10)

    def update_tw():
        while True:
            time.sleep(0.01)
            # tw.append(time.time(), np.random.randn(10))
            tw_plot.append(time.time(), np.random.randn(10))

    update_tw_thread = threading.Thread(target=update_tw, daemon=True)
    update_tw_thread.start()

    app.exec()

    return app



if __name__ == '__main__':

    main()

