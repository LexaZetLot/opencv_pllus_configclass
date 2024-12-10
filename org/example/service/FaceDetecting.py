import os
import random

import cv2

from burp_spring.config.annotation.Autowired import Autowired
from org.example.service.ServiceImpl import ServiceImpl


class FaceDetector(ServiceImpl):
    def __init__(self, training=False):
        self.training = training

        self.face_model = cv2.dnn.readNetFromCaffe('./faces_data/detection/deploy.prototxt',
                                                   './faces_data/detection/res10_300x300_ssd_iter_140000.caffemodel')
        self.face_blob_height = 300
        self.face_average_color = (200, 200, 250)
        self.face_confidence_threshold = 0.9

        self.training_image_size = (300, 300)
        self.imgDAO = None
        self.cap = cv2.VideoCapture(0)

        self.model_detected_face = cv2.face.LBPHFaceRecognizer().create()

    @Autowired()
    def getImgDAO(self, imgDAO):
        self.imgDAO = imgDAO

    def run(self):
        if self.training:
            self.names, self.training_images, self.trainings_labels = self.imgDAO.ReadImgList()
            self.model_detected_face.train(self.training_images, self.trainings_labels)
            path = './' + str(random.randrange(1000000)) + '.xml'
            while os.path.exists(path):
                path = './' + str(random.randrange(1000000)) + '.xml'
            self.model_detected_face.save(path)
            self.imgDAO.InsertXmlSetting(path)
        else:
            self.names = self.imgDAO.NameList()
            path = self.imgDAO.ReadXmlSetting()
            self.model_detected_face.read(path)
            os.remove(path)

        self.ret, self.frame = self.cap.read()
        self.h, self.w = self.frame.shape[:2]
        self.face_blob_width = int(self.face_blob_height * (self.w / self.h))
        self.face_blob_size = (self.face_blob_width, self.face_blob_height)
        while self.ret:
            face_blob = cv2.dnn.blobFromImage(self.frame, size=self.face_blob_size, mean=self.face_average_color)
            self.face_model.setInput(face_blob)
            detections = self.face_model.forward()


            for face in detections[0, 0]:
                face_confidence = face[2]
                if face_confidence > self.face_confidence_threshold:
                    x0, y0, x1, y1 = (face[3:7] * [self.w, self.h, self.w, self.h]).astype(int)
                    y1_roi = y0 + int(1.2 * (y1 - y0))
                    x_margin = ((y1_roi - y0) - (x1 - x0)) // 2
                    x0_roi = x0 - x_margin
                    x1_roi = x1 + x_margin
                    if x0_roi < 0 or x1_roi > self.w or y0 < 0 or y1_roi > self.h:
                        continue

                    cv2.rectangle(self.frame, (x0, y0), (x1, y1), (0, 0, 255), 1)

                    gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                    gray_roi = gray[y0:y1_roi, x0_roi:x1_roi]
                    if gray_roi.shape == 0:
                        continue
                    gray_roi = cv2.resize(gray_roi, self.training_image_size)
                    label, confidence = self.model_detected_face.predict(gray_roi)
                    if confidence < 40.00:
                        text = f'{self.names[label]} confidence={confidence:.2f}'
                    else:
                        text = 'notKnow'
                    cv2.putText(self.frame, text, (x0, y0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

            cv2.imshow('frame', self.frame)
            if cv2.waitKey(1) == 27:
                break
            self.ret, self.frame = self.cap.read()

