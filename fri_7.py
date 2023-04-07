import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QIcon
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from secondwindow import *

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from sort import *

form_class = uic.loadUiType("GUI.ui")[0]


def draw_boxes(img, bbox, identities=None, categories=None, confidences=None, names=None, colors=None):
    # 바운딩 박스
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        tl = int(2) or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        conf = round(confidences[i], 2) if confidences is not None else 0

        color = colors[id]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, tl)

        label = str(id) + ":" + names[cat] + str(
            conf) if identities is not None else f'{names[cat]} {confidences[i]:.2f}'
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

class Ui_MainWindow(QMainWindow,form_class):
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.cap = cv2.VideoCapture()
        self.out = None
        self.timer_video = QtCore.QTimer()
        self.setWindowIcon(QIcon('cloud.png'))
        self.init_slots()
        self.button_video_open()

        source, weights, view_img, save_txt, imgsz, trace = 'shcool_2.mp4', 'yolov7.pt', True, True, int(256), False

        self.device = select_device('')
        self.half = self.device.type != 'cpu'

        cudnn.benchmark = True

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            model = TracedModel(self.model, self.device, int(640))

        if self.half:
            self.model.half()

        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]


    def button_video_open(self):
        video_name = 'shcool_2.mp4'

        if not video_name:
            return

        flag = self.cap.open(video_name)
        if flag == False:
            QtWidgets.QMessageBox.warning(
                self, u"Warning", u"비디오 열기 실패", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.out = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(
                *'MJPG'), 20, (int(self.cap.get(3)), int(self.cap.get(4))))
            self.timer_video.start(30)
            self.SecondWindow.setDisabled(True)
            self.pb.setDisabled(True)

    def show_video_frame(self):
        name_list = []

        flag, img = self.cap.read()
        if img is not None:
            showimg = img
            image = cv2.imread('bird1.jpg')
            yo = image.copy()
            with torch.no_grad():
                img = letterbox(img, new_shape=int(256))[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=True)[0]

                # Apply NMS
                pred = non_max_suppression(pred,0.5, 0.6, classes=0,
                                           agnostic=True)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], showimg.shape).round()

                        dets_to_sort = np.empty((0, 6))
                        # NOTE: We send in detected object class too
                        for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                            dets_to_sort = np.vstack((dets_to_sort,
                                                      np.array([x1, y1, x2, y2, conf, detclass])))

                        tracked_dets = sort_tracker.update(dets_to_sort, True)
                        tracks = sort_tracker.getTrackers()

                        # draw boxes for visualization
                        if len(tracked_dets) > 0:
                            bbox_xyxy = tracked_dets[:, :4]
                            identities = tracked_dets[:, 8]
                            categories = tracked_dets[:, 4]
                            confidences = dets_to_sort[:, 4]
                            for t, track in enumerate(tracks):
                                track_color = self.colors[int(track.detclass)]

                                [cv2.line(showimg, (int(track.centroidarr[i][0]),
                                                int(track.centroidarr[i][1])),
                                          (int(track.centroidarr[i + 1][0]),
                                           int(track.centroidarr[i + 1][1])),
                                          track_color, thickness=int(2))
                                 for i, _ in enumerate(track.centroidarr)
                                 if i < len(track.centroidarr) - 1]
                                [cv2.line(yo, (int(track.centroidarr[i][0] / 2),
                                               int(track.centroidarr[i][1] / 2) + 50),
                                          (int(track.centroidarr[i + 1][0] / 2),
                                           int(track.centroidarr[i + 1][1] / 2) + 50),
                                          track_color, thickness=int(2))
                                 for i, _ in enumerate(track.centroidarr)
                                 if i < len(track.centroidarr) - 1]

                        showimg = draw_boxes(showimg, bbox_xyxy, identities, categories, confidences, self.names, self.colors)
                        '''
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            print(label)
                            plot_one_box(
                                xyxy, showimg, label=label, color=self.colors[int(cls)], line_thickness=2)
                        '''

            self.out.write(showimg)
            self.out.write(yo)
            show1 = cv2.resize(showimg, (640, 480))
            show2 = cv2.resize(yo,(640,480))
            show = np.hstack((show1,show2))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))


        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label.clear()
            self.SecondWindow.setDisabled(False)
            self.pb.setDisabled(False)

    def init_slots(self):
        self.SecondWindow.clicked.connect(self.button_Second)
        self.timer_video.timeout.connect(self.show_video_frame)



    def button_Second(self):
        self.second = secondWindow()
        self.second.exec()
        self.show()

sort_tracker = Sort(max_age=5, min_hits=2,iou_threshold=0.2)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())

