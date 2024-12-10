import cv2

from burp_spring.config.annotation.Autowired import Autowired
from org.example.service.ServiceImpl import ServiceImpl


class FaceGenerator(ServiceImpl):
    def __init__(self, nameIdFace='np'):
        self.nameIdFace = nameIdFace

        self.face_model = cv2.dnn.readNetFromCaffe('./faces_data/detection/deploy.prototxt',
                                                   './faces_data/detection/res10_300x300_ssd_iter_140000.caffemodel')
        self.face_blob_height = 300
        self.face_average_color = (104, 177, 123)
        self.face_confidence_threshold = 0.9

        self.cap = cv2.VideoCapture(0)
        self.ret, self.frame = self.cap.read()
        print()
        self.h, self.w = self.frame.shape[:2]
        face_blob_width = int(self.face_blob_height * (self.w / self.h))
        self.face_blob_size = (face_blob_width, self.face_blob_height)
        self.imgDAO = None

    @Autowired()
    def getImgDAO(self, imgDAO):
        self.imgDAO = imgDAO

    def run(self):

        imgList = []
        count = 0
        while self.ret and count < 300:
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

                    gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                    face_img = cv2.resize(gray[y0:y1_roi, x0_roi:x1_roi], (300, 300))
                    imgList.append(face_img)
                    count += 1
                    print(count)

            cv2.imshow('frame', self.frame)
            if cv2.waitKey(1) == 27:
                break
            self.ret, self.frame = self.cap.read()
        self.imgDAO.InsertImgList(self.nameIdFace, imgList)
