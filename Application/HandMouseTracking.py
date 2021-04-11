import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

import imutils as imutils
import numpy as np
from plyer import notification

cascPath = "HandMouseCascade.xml"
handMouseCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log', level=log.INFO)

video_capture = cv2.VideoCapture(0)

anterior = 0
time = dt.datetime.min

timeDelay = 10
enableNotif = True

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture webcam output
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(frame, 20, 30)

    # edges_high_thresh = cv2.Canny(blur, 60, 120)
    # edges_high_thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    # edges_high_thresh = cv2.erode(edges_high_thresh, None, iterations=2)
    # edges_high_thresh = cv2.dilate(edges_high_thresh, None, iterations=2)
    # cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # c = max(cnts, key=cv2.contourArea)
    # extLeft = tuple(c[c[:, :, 0].argmin()][0])
    # extRight = tuple(c[c[:, :, 0].argmax()][0])
    # extTop = tuple(c[c[:, :, 1].argmin()][0])
    # extBot = tuple(c[c[:, :, 1].argmax()][0])
    # cv2.drawContours(frame, [c], -1, (0, 255, 255), 2)
    # cv2.circle(frame, extLeft, 8, (0, 0, 255), -1)
    # cv2.circle(frame, extRight, 8, (0, 255, 0), -1)
    # images = np.hstack((edges, fgmask))
    # cv2.imshow('Frame', edges_high_thresh)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hands = handMouseCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minSize=(200, 200)
    )

    # Draw a rectangle around the hands
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        objectHand = frame[x:x + w, y:y + h]
        objectHand = np.array(objectHand)
        grayObject = cv2.cvtColor(objectHand, cv2.COLOR_BGR2GRAY)
        circle = cv2.circle(grayObject, (int((x + w)/2), int((y + h)/2)), 8, (255, 255, 255))
        cv2.imshow('Frame', grayObject)

    if anterior != len(hands):
        anterior = len(hands)
        log.info("faces: " + str(len(hands)) + " at " + str(dt.datetime.now()))

        duration = dt.datetime.now() - time
        if duration.total_seconds() > timeDelay and enableNotif:
            time = dt.datetime.now()

            hour = time.hour
            minute = time.minute
            second = time.second

            notification.notify(
                title="Wrist Watch",
                message="Your wrist is in an unhealthy position!\n" + str(hour) + ":" + str(minute) + ":" + str(second),
                app_icon="CitrusCircle.ico",
                timeout=60  # Notification lasts a minute
            )

        duration = dt.datetime.now() - time
        if duration.total_seconds() > timeDelay:
            time = dt.datetime.now()
            notification.notify(
                title="Wrist Watch",
                message="Your wrist is in an unhealthy position!",
                timeout=60  # Notification lasts a minute
            )

    # Display
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display
    cv2.imshow('Video', frame)

video_capture.release()
cv2.destroyAllWindows()
