"""
2024/8/10 20:52
本文件由my_ywj首次创建编写
"""
import json
import os
import subprocess

from PyQt5 import QtGui

from test import PolygonLabel

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
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

from main_win.multi_win import Ui_MainWindow
from ultralytics import YOLO


class DetectThread(QThread):
    """
    the thread for detect
    """
    index_img = pyqtSignal(int, np.ndarray)

    def __init__(self, index: int, source: str, judge):
        """
        index: the number of show window
        source: model input source
        judge: diffent logit to judge box
        """
        super(DetectThread, self).__init__()
        print(f"{index}")
        self.index = index
        print(f"init model--{self.index}")
        self.model = YOLO("yolov8n.pt")
        print(f"init model over")
        self.path = source
        self.use_warn = False
        self.points = []
        self.new_points = []
        self.judge = judge

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
                self.judge_result(results[0])
                self.annotated_frame = results[0].plot()
                im0 = self.annotated_frame[..., ::-1]
                im0 = self.set_mask_frame(im0)
                self.index_img.emit(self.index, im0)
                # Display the annotated frame
            else:
                # Break the loop if the end of the video is reached
                break

    def set_warn_area(self, points=None):
        """
        draw danger box
        :return:
        """
        # points = [(3, 2), (633, 6), (630, 474), (6, 472)]
        # 这些点是在(640,480)图像上绘制的，需要把坐标转换一下到当前图片上

        self.points = points
        print("points", self.points)
        self.new_points = []
        # for point in points:
        #     x, y = point
        #     new_x = int(x * (self.annotated_frame.shape[0] / 480))
        #     new_y = int(y * (self.annotated_frame.shape[1] / 640))
        #     self.mask_frame = self.mask_frame.resize(self.annotated_frame.shape[:2])
        #     cv2.circle(self.mask_frame, (new_x, new_y), 5, (255, 0, 0), -1)
        print("points", points)

    def set_mask_frame(self, img_array):
        """
        set mask frame
        :param img_array:
        :return:
        """

        if len(self.new_points)!=0:
            mat_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            cv2.polylines(mat_img, [np.array(self.new_points, dtype=np.int32)], True, (0, 0, 255), 2)
            return mat_img

        if self.points is not None:
            for point in self.points:
                x, y = point
                new_x = int(x * (self.annotated_frame.shape[1] / 640))
                new_y = int(y * (self.annotated_frame.shape[0] / 480))
                self.new_points.append((new_x, new_y))
                # 将new_points中所有点连成的线画到img_array上
            mat_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            cv2.polylines(mat_img, [np.array(self.new_points, dtype=np.int32)], True, (0, 0, 255), 2)
            img_array = mat_img
        return img_array

    def judge_result(self, results):
        """

        :param results: yolo results
        :return:
        """

        if self.judge == 1:
            self.judge_in(results)
        elif self.judge == 2:
            self.judge_out(results)
        elif self.judge == 3:
            self.count_area(results)
        elif self.judge == 4:
            self.count_num(results)

    def count_num(self, results, single=False):
        """
        count num for every boxs
        :param results:
        :param single: if single is False, count area for every boxs, if single is True, count area for single box.
        ex. if you want to count area for inside area, set single=True,and let other alg invoking it.
        :return: all classes name and number of classes
        """
        num_cls = len(results)
        cls_dict = {v: 0 for k, v in results.names.items()}
        for result in results:
            if len(result) > 0:
                x0 = result.boxes.xyxy[0][0].item()
                y0 = result.boxes.xyxy[0][1].item()
                x1 = result.boxes.xyxy[0][2].item()
                y1 = result.boxes.xyxy[0][3].item()
                center_xy = ((x0 + x1) / 2, (y0 + y1) / 2)
                if self.is_point_in_polygon(center_xy, self.new_points):
                    print("inside：num+1")
                    cls_name = result.names.get(result.boxes.cls.item())
                    cls_dict[cls_name] += 1

        for k, v in cls_dict.items():
            if v == 0:
                continue
            print(f"类别{k},有{v}个。")
        return cls_dict

    def count_area(self, results, single=False):
        """
        compute area for every boxs
        :param results:
        :param single: if single is False, count area for every boxs, if single is True, count area for single box.
        ex. if you want to count area for inside area, set single=True,and let other alg invoking it.
        :return:
        """
        if self.new_points is not None:
            if single:
                results = [results]
            for result in results:
                if len(result) > 0:
                    x0 = result.boxes.xyxy[0][0].item()
                    y0 = result.boxes.xyxy[0][1].item()
                    x1 = result.boxes.xyxy[0][2].item()
                    y1 = result.boxes.xyxy[0][3].item()
                    center_xy = ((x0 + x1) / 2, (y0 + y1) / 2)
                    if self.is_point_in_polygon(center_xy, self.new_points):
                        print("inside：compute area")
                        try:
                            area = (x1 - x0) * (y1 - y0)
                            cls_name = result.names.get(result.boxes.cls.item())
                            print(f"当前检测到{cls_name}，面积：{area}")
                        except:
                            print("没有检测到目标")

    def judge_in(self, results):
        if self.new_points is not None:
            for result in results:
                if len(result) > 0:
                    x0 = result.boxes.xyxy[0][0].item()
                    y0 = result.boxes.xyxy[0][1].item()
                    x1 = result.boxes.xyxy[0][2].item()
                    y1 = result.boxes.xyxy[0][3].item()
                    center_xy = ((x0 + x1) / 2, (y0 + y1) / 2)

                    if self.is_point_in_polygon(center_xy, self.new_points):
                        print("inside")

    def judge_out(self, results):
        if self.new_points is not None:
            for result in results:
                if len(result) > 0:
                    x0 = result.boxes.xyxy[0][0].item()
                    y0 = result.boxes.xyxy[0][1].item()
                    x1 = result.boxes.xyxy[0][2].item()
                    y1 = result.boxes.xyxy[0][3].item()
                    center_xy = ((x0 + x1) / 2, (y0 + y1) / 2)

                    if not self.is_point_in_polygon(center_xy, self.new_points):
                        print("outside")


    def is_point_in_polygon(self, point, points):
        x, y = point
        n = len(points)
        inside = False
        if n>0:
            p1x, p1y = points[0]
            for i in range(n + 1):
                p2x, p2y = points[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside

                p1x, p1y = p2x, p2y

            # if inside:
            #     print("inside")
            # else:
            #     print("outside")
            return inside


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.configs = json.load(open('config/configs.json', 'r'))

        self.init_ui()
        self.init_var()
        self.init_action()
        self.init_detect()

    def init_ui(self):
        # self.toolbox.hide()
        # self.operation_box.hide()
        self.foot_box.hide()
        self.gridLayout.alignment()
        self.draw_label = PolygonLabel()
        # set self.draw_label width and height
        self.draw_label.setFixedSize(640, 480)

    def init_action(self):
        self.up_btn.clicked.connect(self.up_btn_action)
        self.down_btn.clicked.connect(self.down_btn_action)
        self.warn_btn.clicked.connect(self.set_warn_box)
        self.draw_label.send_points.connect(self.send_points)
        self.label_btn.clicked.connect(self.label_btn_action)

    def init_var(self):

        self.titles = []

        self.scale = 0  # control the number of windows pages.

        self.detect1 = QThread()
        self.detect2 = QThread()
        self.detect3 = QThread()
        self.detect4 = QThread()
        self.detect5 = QThread()
        self.detect6 = QThread()
        self.detect7 = QThread()
        self.detect8 = QThread()
        self.detect9 = QThread()
        self.detect10 = QThread()
        self.detect11 = QThread()
        self.detect12 = QThread()
        self.detect13 = QThread()
        self.detect14 = QThread()
        self.detect15 = QThread()
        self.detect16 = QThread()
        self.detect17 = QThread()
        self.detect18 = QThread()
        self.detect19 = QThread()
        self.detect20 = QThread()
        self.detect21 = QThread()
        self.detect22 = QThread()
        self.detect23 = QThread()
        self.detect24 = QThread()
        self.detect25 = QThread()
        self.detect26 = QThread()
        self.detect27 = QThread()
        self.detect28 = QThread()
        self.detect30 = QThread()
        self.detect31 = QThread()
        self.detect32 = QThread()

        self.threads=[
            self.detect1,
            self.detect2,
            self.detect3,
            self.detect4,
            self.detect5,
            self.detect6,
            self.detect7,
            self.detect8,
            self.detect9,
            self.detect10,
            self.detect11,
            self.detect12,
            self.detect13,
            self.detect14,
            self.detect15,
            self.detect16,
            self.detect17,
            self.detect18,
            self.detect19,
            self.detect20,
            self.detect21,
            self.detect22,
            self.detect23,
            self.detect24,
            self.detect25,
            self.detect26,
            self.detect27,
            self.detect28,
            self.detect30,
            self.detect31,
            self.detect32,
        ]

        self.labels = [self.win1_label, self.win2_label, self.win3_label, self.win4_label]

    def init_detect(self) -> None:
        for index, json_data in enumerate(self.configs, start=1):
            self.init_win(index, json_data.get('source'), json_data.get('judge',0))
            print(f'初始化{index}_{json_data.get("source")}')
            self.warn_comboBox.addItem(json_data.get('name'))  # 顺道把comboBox的选项添加进去
            self.titles.append(json_data.get('name'))
            self.send_points(json_data.get('points'), index)
        self.init_lable()

    def init_lable(self):
        self.win1_label.clear()
        self.win2_label.clear()
        self.win3_label.clear()
        self.win4_label.clear()
        try:
            self.title1_label.setText(self.titles[4*self.scale])
        except:
            self.title1_label.clear()

        try:
            self.title2_label.setText(self.titles[4 * self.scale+1])
        except:
            self.title2_label.clear()

        try:
            self.title3_label.setText(self.titles[4 * self.scale+2])
        except:
            self.title3_label.clear()

        try:
            self.title4_label.setText(self.titles[4 * self.scale+3])
        except:
            self.title4_label.clear()


    def init_win(self, index, path, judge) -> None:
        """
        init base func,and set slot
        :param index:
        :param path:
        :return:
        """
        self.threads[index] = DetectThread(index, path, judge)
        self.threads[index].start()
        self.threads[index].index_img.connect(self.result_util)
        self.threads[index].index_img.connect(self.draw_rectangle)
        print(f"init over {index, path}")

    def result_util(self, index, img_array):
        if 4 * self.scale < index <= 4 * (self.scale+1):
            index = index - 4 * self.scale
            try:
                threading.Thread(target=self.thread_result_util, args=[self.labels[index-1], img_array]).start()
            except Exception as e:
                print('result_util' + str(e))

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
            show_img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(show_img))

        except Exception as e:
            print(repr(e))

    def up_btn_action(self):
        if self.scale == 0:
            print("当前已经是第一屏")
            return
        self.scale -= 1
        self.init_lable()

    def down_btn_action(self):
        if (self.scale + 1) * 4 > len(self.configs):
            print("当前已到最大屏")
            return
        self.scale += 1
        self.init_lable()

    def set_warn_box(self):
        """
        set warning box
        :return:
        """

        self.draw_label.clear_polygon_points()
        self.draw_label.show()


    def draw_rectangle(self, index, img_array):
        # warn_show_shape = self.threads[index].frame_shape
        # # 创创建一个既能显示图片又能画线的QLabel
        # 将图片显示到draw_label上
        if index == self.warn_comboBox.currentIndex()+1:
            ih, iw, _ = img_array.shape
            img_src_ = cv2.resize(img_array, (640, 480))
            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            show_img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                              QImage.Format_RGB888)
            self.draw_label.setPixmap(QPixmap.fromImage(show_img))

            print(index)

    def send_points(self, qpoint_list, index=None):
        if qpoint_list is None:
            qpoint_list=[]
        try:
            coordinate_list = [(point.x(), point.y()) for point in qpoint_list]
        except:
            coordinate_list = [(point[0], point[1]) for point in qpoint_list]
        cur_index = self.warn_comboBox.currentIndex()
        print("cur_index", cur_index)
        if index is not None:
            self.threads[index].set_warn_area(coordinate_list)
        else:
            self.threads[self.warn_comboBox.currentIndex() + 1].set_warn_area(coordinate_list)
            self.configs[cur_index]["points"] = coordinate_list

        # print("points", coordinate_list)

    def label_btn_action(self):
        try:
            labelme_path = "D:/software/anaconda3/envs/learn/Scripts/labelImg.exe"
            img_path = ""
            command = [labelme_path, img_path]
            subprocess.Popen(command)
        except Exception as e:
            print(str(e))

    def closeEvent(self, event: QtGui.QCloseEvent):
        reply = QMessageBox.question(self, '确认退出', '确定要退出吗？', QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 用户点击了确认按钮，执行退出操作
            try:
                with open("config/configs.json", "w") as f:
                    json.dump(self.configs, f)
            except:
                print("error")
            event.accept()
            os._exit(0)
        else:
            # 用户点击了取消按钮，忽略关闭事件
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    # myWin.showMaximized()
    sys.exit(app.exec_())