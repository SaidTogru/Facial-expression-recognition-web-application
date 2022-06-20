import cv2
import os
import os

path_utils = "GoogleAutomation//utils_for_collecting//"
facec = cv2.CascadeClassifier(path_utils+'haarcascade_frontalface_default.xml')


for path, subdirs, files in os.walk("jaffe"):
    for name in files:
        imgpath = os.path.join(path, name)
        fr = cv2.imread(imgpath)
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            cv2.imwrite(imgpath, fc)
