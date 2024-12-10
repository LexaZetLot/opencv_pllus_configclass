import cv2
import numpy as np
import pyautogui
import mediapipe as mp
import time
import os
import math



cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False, max_num_hands=1)
npDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

point0 = 0
point4 = ()
point4Tap = ()
point7 = ()
point8 = ()
lenPhalanx = 0
lenTap = 0
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 4:
                    cxSize, cySize = int(round(lm.x, 3) * 1920), int(round(lm.y, 3) * 1080)
                    pyautogui.moveTo(cxSize, cySize, _pause=False)
                    point4 = (cx, cy)
                    point4Tap = (cxSize, cySize)
                elif id == 7:
                    point7 = (cx, cy)
                elif id == 8:
                    point8 = (cx, cy)
                elif id == 0:
                    point0 = (cx, cy)


                if id == 8 or id == 12 or id == 4:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)


            lenPhalanx = math.sqrt((point0[0] - point4[0]) ** 2 + (point0[1] - point4[1]) ** 2)
            lenTap = math.sqrt((point8[0] - point4[0]) ** 2 + (point8[1] - point4[1]) ** 2)


            if lenPhalanx * 0.1 > lenTap:
                print(point4)
                pyautogui.click(point4Tap[0], point4Tap[1], _pause=False)
            npDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cv2.imshow('python', img)
    if cv2.waitKey(20) == 27:
        break