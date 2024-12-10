import numpy as np
import cv2

from org.example.service.ServiceImpl import ServiceImpl


class Tracker(ServiceImpl):
    def __init__(self):
        self.x0, self.y0, self.x1, self.y1 = 0, 0, 0, 0
        self.new_roi = False
        self.cat = cv2.VideoCapture(0)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.drawFunction)
        self.ret, self.frame = self.cat.read()
        self.h, self.w = self.frame.shape[:2]

    def drawFunction(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x0, self.y0, = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.x1, self.y1 = x, y
            self.new_roi = True

    def run(self):
        while True:
            ret, frame = self.cat.read()

            if self.x0 != 0 and self.y0 != 0 and self.x1 != 0 and self.y1 != 0:
                if self.x0 < 0 or self.x1 > self.w or self.y0 < 0 or self.y1 > self.h:
                    continue
                if self.new_roi:
                    roi = frame[self.y0:self.y1, self.x0:self.x1]
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    hist_roi = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
                    cv2.normalize(hist_roi, hist_roi, 0, 255, cv2.NORM_MINMAX)
                    track_window = (self.x0, self.y0, self.x1, self.y1)
                    self.new_roi = False

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                back_proj = cv2.calcBackProject([hsv], [0], hist_roi, [0, 180], 1)
                rotated_rect, track_window = cv2.CamShift(back_proj, track_window, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1))

                box_points = cv2.boxPoints(rotated_rect)
                box_points = np.int64(box_points)
                cv2.polylines(frame, [box_points], True, (255, 0, 0), 2)

            cv2.imshow('image', frame)
            if cv2.waitKey(1) == 27:
                break
