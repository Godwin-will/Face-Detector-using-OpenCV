import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)

face_detect = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_detect = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGRA2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        faces = face_detect.detectMultiScale(roi_gray, 1.3, 5)
        for (ex,ey,ew,eh) in faces:
            cv.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,0,0), 3)
            
    cv.imshow("detection", frame)
        
    if cv.waitKey(1) == ord('q'):
        break
    
capture.release()
cv.destroyAllWindows()