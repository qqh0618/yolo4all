"""
2024/8/10 20:52
本文件由my_ywj首次创建编写
"""
"""
2024/8/6 11:45
本文件由my_ywj首次创建编写
"""
import sys
import threading

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow

from main_win.multi_win import Ui_MainWindow
from ultralytics import YOLO


class DetectThread(QThread):
    """
    the thread for detect
    """
    index_img = pyqtSignal(int, np.ndarray)

    def __init__(self, index: int, source: str):
        """
        index: the number of show window
        source: model input source
        """
        super(DetectThread, self).__init__()
        print(f"{index}")
        self.index = index
        print(f"init model--{self.index}")
        self.model = YOLO("yolov8n.pt")
        print(f"init model over")
        self.path = source

    def run(self) -> None:
        print(f"predict source of {self.path}")
        cap = cv2.VideoCapture(self.path)

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 inference on the frame
                results = self.model(frame)
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                im0 = annotated_frame[..., ::-1]
                self.index_img.emit(self.index, im0)
                # Display the annotated frame
            else:
                # Break the loop if the end of the video is reached
                break


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.labels = [self.win1_label, self.win2_label, self.win3_label, self.win4_label]
        self.init_detect()

    def init_detect(self) -> None:
        self.init_win1(1, "1.mp4")
        self.init_win2(2, "2.mp4")
        self.init_win3(3, "3.mp4")
        self.init_win4(4, "4.mp4")
        print("初始化")

    def init_win1(self, index, path) -> None:
        self.detect1 = DetectThread(index, path)
        self.detect1.start()
        self.detect1.index_img.connect(self.result_util)
        print(f"init over {index, path}")

    def init_win2(self, index, path) -> None:
        self.detect2 = DetectThread(index, path)
        self.detect2.start()
        self.detect2.index_img.connect(self.result_util)
        print(f"init over {index, path}")

    def init_win3(self, index, path) -> None:
        self.detect3 = DetectThread(index, path)
        detect3_thread = QThread()
        # detect3.moveToThread(detect3_thread)
        self.detect3.start()
        self.detect3.index_img.connect(self.result_util)
        print(f"init over {index, path}")

    def init_win4(self, index, path) -> None:
        self.detect4 = DetectThread(index, path)
        # detect4_thread = QThread()
        # detect4.moveToThread(detect4_thread)
        self.detect4.start()
        self.detect4.index_img.connect(self.result_util)
        print(f"init over {index, path}")

    def result_util(self, index, img_array):
        try:
            threading.Thread(target=self.thread_result_util, args=[self.labels[index-1], img_array]).start()
        except Exception as e:
            print(str(e))

    def thread_result_util(self, label, img_src):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))



if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    # myWin.showMaximized()
    sys.exit(app.exec_())