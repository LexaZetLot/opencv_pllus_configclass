from org.example.service.Tracking import Tracker
from org.example.service.TrackingKNN import TrackingKNN
from burp_spring.config.annotation.Bean import Bean
from org.example.service.FaceDetecting import FaceDetector
from org.example.service.FaceGenerate import FaceGenerator
from org.example.dao.ImgDAO import ImgDAO
from org.example.service.YoloFace import YoloFace


class Config:
    @Bean()
    def getImgDAO(self):
        return ImgDAO()

    @Bean(nameBean='faceDetecting')
    def getFaceDetecting(self):
        return FaceDetector()

    @Bean()
    def getFaceGenerator(self):
        return FaceGenerator()

    @Bean()
    def getTracking(self):
        return Tracker()

    @Bean()
    def getTrackingKNN(self):
        return TrackingKNN()

    @Bean()
    def getYoloFace(self):
        return YoloFace()