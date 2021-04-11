import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "HandMouseCascade.xml"
handMouseCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture webcam output
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hands = handMouseCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minSize=(150, 150)
    )

    # Draw a rectangle around the hands
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(hands):
        anterior = len(hands)
        log.info("faces: "+str(len(hands))+" at "+str(dt.datetime.now()))

    # Display
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display
    cv2.imshow('Video', frame)

video_capture.release()
cv2.destroyAllWindows()
