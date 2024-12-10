import cv2

from org.example.service.ServiceImpl import ServiceImpl


class TrackingKNN(ServiceImpl):
    def __init__(self):
        self.bgSubtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)
        self.erodeKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
        self.dilateKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 11))
        self.cap = cv2.VideoCapture(0)

    def run(self):
        ret, frame = self.cap.read()
        while ret:
            fgMask = self.bgSubtractor.apply(frame)

            _, thresh = cv2.threshold(fgMask, 244, 255, cv2.THRESH_BINARY)
            cv2.erode(thresh, self.erodeKernel, thresh, iterations=2)
            cv2.dilate(thresh, self.dilateKernel, thresh, iterations=2)

            contours, heir = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) > 1000:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == 27:
                break
            ret, frame = self.cap.read()