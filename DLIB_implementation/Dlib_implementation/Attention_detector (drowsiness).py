import cv2 as cv
import numpy as np
import dlib
import imutils
import glob
from scipy.spatial import distance as dist
from imutils import face_utils
from threading import Thread
from gtts import gTTS
from playsound import playsound
import time
counter=0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector1 = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
class DrowsinessDetector:
    def __init__(self,shape):
        self.shape=shape
        counter=0
    def eye_aspect_ratio(self,eye):
        A = np.linalg.norm(eye[1]-eye[5])
        B = np.linalg.norm(eye[2]-eye[4])
        C = np.linalg.norm(eye[0]-eye[3])
        EAR= (A+B)/(2.0*C)
        return EAR
    def final_ear(self):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = self.shape[lStart:lEnd]
        rightEye = self.shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        ear = (leftEAR+rightEAR)/2.0
        return (ear,leftEye,rightEye)
    def alert(self,minm_EAR=0.3,EAR_threshold=30): #minm_EAR-if person eye aspect ratio fall beyond it then person eyes are closed 
        global counter
        
        alarm_status=False                                   #if person eyes are closed for more than EAR_threshold then alarm will begin to ring 
        eye = self.final_ear()
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]
        if ear < minm_EAR:
            counter += 1

            if counter >= EAR_threshold:
                if alarm_status == False:
                    alarm_status = True
        else:
            counter= 0
            alarm_status = False
        return alarm_status

        












































#cvdetector = cv.CascadeClassifier("https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml")




video=cv.VideoCapture(0)
print("Hold on, starting webcam")
seconds=0
while True:
    second=time.time()
    ret,img=video.read()  

    

    if(ret):
        #print("OK")
        gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        rects = detector1.detectMultiScale(gray_img, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30),flags=cv.CASCADE_SCALE_IMAGE)
       
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
            shape = predictor(gray_img, rect)
            
            shape = face_utils.shape_to_np(shape)
            drowsiness=DrowsinessDetector(shape)
            alarm_status=drowsiness.alert()
            print(counter)
            if(alarm_status):
                seconds+=time.time()-second
                cv.putText(img, "DROWSINESS ALERT!", (10, 30),cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        
        cv.putText(img,str(seconds),(30,50),cv.FONT_HERSHEY_TRIPLEX,0.8,(0,0,255),2)
        cv.imshow("Head Pose",img)
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break
    else:
        print("Webcam not found or it is being used by another application")
        exit(0)
        
video.release()
cv.destroyAllWindows()

