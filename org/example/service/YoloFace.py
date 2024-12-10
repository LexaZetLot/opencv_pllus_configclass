import cv2
from ultralytics import YOLO


from org.example.service.ServiceImpl import ServiceImpl


class YoloFace(ServiceImpl):
    def __init__(self):
        self.yolo = YOLO('yolo11n.pt').load()
        self.cap = cv2.VideoCapture(0)

    def run(self) -> None:
        pedestrians = []
        while True:
            ret, frame = self.cap.read()
            result = self.yolo.track(frame, stream=True)

            for res in result:
                className = res.names
                for box in res.boxes:
                    [x1, y1, x2, y2] = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls = int(box.cls[0])
                    if className[cls] == 'person':
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, className[cls], (x1, y1 + 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)


            cv2.imshow('YoloFace', frame)
            if cv2.waitKey(1) == 27:
                break
