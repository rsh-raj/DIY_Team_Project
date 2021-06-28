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

counter = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector1 = cv.CascadeClassifier("haarcascade_frontalface_default.xml")


class DrowsinessDetector:
    def __init__(self, shape):
        self.shape = shape
        counter = 0

    def eye_aspect_ratio(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        EAR = (A + B) / (2.0 * C)
        return EAR

    def final_ear(self):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = self.shape[lStart:lEnd]
        rightEye = self.shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        return (ear, leftEye, rightEye)

    def alert(self,minm_EAR=0.3,EAR_threshold=30):  #minm_EAR-if person eye aspect ratio fall beyond it then person eyes are closed
        global counter

        alarm_status = False  #if person eyes are closed for more than EAR_threshold then alarm will begin to ring
        eye = self.final_ear()
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]
        if ear < minm_EAR:
            counter += 1

            if counter >= EAR_threshold:
                if alarm_status == False:
                    alarm_status = True
        else:
            counter = 0
            alarm_status = False
        return alarm_status
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30,0.001)

objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)  #numpy method to store coordinates of a grid

objpoints = []  # basically objpoints store coordinate of chessboard corners in real 3d world
imgpoints = []  # 2d points in image plane.

images = glob.glob('data/images/*.jpg')



for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

    if ret:
        objpoints.append(objp)
                #objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                           criteria)
        imgpoints.append(corners2)

        img = cv.drawChessboardCorners(img, (7, 6), corners2, ret)

        cv.imshow('img', img)
        print(f"fuck")
        cv.waitKey(500)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
cv.destroyAllWindows()
        
class HeadposeYawn:
    def __init__(self, img,mtx):
        self.img = img
        self.mtx=mtx

    def cam_callibration(self):
        pass


    def ref3DModel(self):
        modelPoints = [[0.0, 0.0, 0.0], [0.0, -330.0, -65.0],
                       [-255.0, 170.0, -135.0], [225.0, 170.0, -135.0],
                       [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]]
        return np.array(modelPoints, dtype=np.float64)

    def ref2DImagePoints(self, shape):
        imagePoints = [[shape.part(30).x, shape.part(30).y],
                       [shape.part(8).x, shape.part(8).y],
                       [shape.part(36).x, shape.part(36).y],
                       [shape.part(45).x, shape.part(45).y],
                       [shape.part(48).x, shape.part(48).y],
                       [shape.part(54).x, shape.part(54).y]]
        return np.array(imagePoints, dtype=np.float64)

    def lip_distance(shape):
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))

        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))

        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)

        distance = abs(top_mean[1] - low_mean[1])
        return distance

    def drawPolyline(self, img, shapes, start, end, isClosed=False):
        points = []
        for i in range(start, end + 1):
            point = [shapes.part(i).x, shapes.part(i).y]
            points.append(point)
        points = np.array(points, dtype=np.float32)
        cv.polylines(img,
                     np.int32([points]),
                     isClosed, (255, 80, 0),
                     thickness=1,
                     lineType=cv.LINE_8)

    def draw(self, img, shapes):
        self.drawPolyline(img, shapes, 0, 16)
        self.drawPolyline(img, shapes, 17, 21)
        self.drawPolyline(img, shapes, 22, 26)
        self.drawPolyline(img, shapes, 27, 30)
        self.drawPolyline(img, shapes, 30, 35, True)
        self.drawPolyline(img, shapes, 36, 41, True)
        self.drawPolyline(img, shapes, 42, 47, True)
        self.drawPolyline(img, shapes, 48, 59, True)
        self.drawPolyline(img, shapes, 60, 67, True)

    def headpose_estimator(self):
        faces = detector(cv.cvtColor(self.img, cv.COLOR_BGR2RGB), 0)
        face3Dmodel = self.ref3DModel()
        
        for face in faces:
            shape = predictor(cv.cvtColor(self.img, cv.COLOR_BGR2RGB), face)

            self.draw(self.img, shape)

            shape1 = face_utils.shape_to_np(shape)

            distance = self.lip_distance(shape1)

            refImgPts = self.ref2DImagePoints(shape)

            height, width, channels = (self.img).shape

            mdists = np.zeros((4, 1), dtype=np.float64)

            success, rotationVector, translationVector = cv.solvePnP(
                face3Dmodel, refImgPts, self.mtx, mdists)

            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv.projectPoints(
                noseEndPoints3D, rotationVector, translationVector, self.mtx,
                mdists)

            rmat, jac = cv.Rodrigues(rotationVector)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)
        return angles




video = cv.VideoCapture(0)
print("Hold on, starting webcam")
while True:
    ret, img = video.read()
    GAZE = "FACE not found"

    if (ret):
        #Drowsiness part
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        rects = detector1.detectMultiScale(gray_img,
                                           scaleFactor=1.1,
                                           minNeighbors=5,
                                           minSize=(30, 30),
                                           flags=cv.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            shape = predictor(gray_img, rect)

            shape = face_utils.shape_to_np(shape)
            drowsiness = DrowsinessDetector(shape)
            alarm_status = drowsiness.alert()
            print(counter)
            if (alarm_status):
                cv.putText(img, "DROWSINESS ALERT!", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        headpose_yawn = HeadposeYawn(img,mtx)
        #headpose_yawn.cam_callibration()
        angles = headpose_yawn.headpose_estimator()
       # angles=[15,19]
        if angles[1] < -15:
            GAZE = "Looking: Left"
        elif angles[1] > 15:
            GAZE = "Looking: Right"
        else:
            GAZE = "Forward"
        cv.putText(img, GAZE, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 255, 80), 2)

        cv.imshow("Head Pose", img)
        key = cv.waitKey(1) & 0xFF
        if key == 27:
            break
    else:
        print("Webcam not found or it is being used by another application")
        exit(0)

video.release()
cv.destroyAllWindows()
