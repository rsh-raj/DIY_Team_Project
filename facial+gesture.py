
import cv2
import numpy as np
import mediapipe as mp
import math
import face_recognition
from threading import Thread
from gtts import gTTS
from playsound import playsound
import glob
import time
import json
from datetime import datetime
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
#Attendance Part
attendanceList = []
def makeAttendanceEntry(name,file_name):
    try:
        with open(file_name,'r+') as FILE:
            allLines = FILE.read().splitlines()
            for line in allLines:
                entry = line.split(',')
                attendanceList.append(entry[0])
            if (name not in attendanceList) and (name != "Unknown") :
                now = datetime.now()
                dtString = now.strftime('%d/%b/%Y, %H:%M:%S')
                FILE.writelines(f'\n{name},{dtString}')
    except:
        print("Attendance list file doesn't exist")

with open("Face_encodings",'r') as f:
    known_face_encodings=json.load(f)
with open("Known_face_names",'r') as f:
    known_face_names=json.load(f)
for list in known_face_encodings:
    list=np.array(list)
known_face_encodings=np.array(known_face_encodings)

file_name=input("Enter the name of file in which you want to take the attendance: ")

if(file_name[-4:]!=".csv"):
    file_name+=".csv"
with open(file_name,'w') as f:
    pass

cap = cv2.VideoCapture(0)
while True:
    flag=False
    check, unknown_image=cap.read()
    if check:
        small_image=cv2.resize(unknown_image,(0,0),None,0.25,0.25)
        small_image=cv2.cvtColor(unknown_image,cv2.COLOR_BGR2RGB)
    
        face_locations = face_recognition.face_locations(small_image)
        face_encodings = face_recognition.face_encodings(small_image, face_locations)

        

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                makeAttendanceEntry(name,file_name)
            cv2.rectangle(unknown_image,(left, top), (right, bottom), (0,255,0),3 )
    
            print(name)
            
            cv2.putText(unknown_image,name,(left+6,top-6), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2,cv2.LINE_AA)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                makeAttendanceEntry(name,file_name)
                #cv2.putText(unknown_image,"Match found, Proceeding to attention detection part",(left+10,top-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2,cv2.LINE_AA)
                flag=True
        cv2.imshow("", unknown_image)
        cv2.waitKey(1)
        if(flag):
           # time.sleep(20)
            break

        
    else:
        print("Failed to load webcam")
        break
cap.release()
cv2.destroyAllWindows()
time.sleep(1)
#Gesture control part
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
            (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        
        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

detector=handDetector(detectionCon=0.8)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
area = 0
colorVol = (255, 0, 0)
















#Facial features recognition part
mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

def cam_callibration():
    '''Returns the two intrinsic camera matrix and distortion matrix of the calibrated camera'''
# termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0: 7, 0: 6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('data/images/*.jpg')

    for image in images: #standard method to callibrate camera by finding chess board corner in an image
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return (mtx,dist)
        
def ref3DModel():
    '''Returns the coordinates of certain facial landmarks of a generic 3d model in a numpy array'''
    modelPoints = [[0.0, 0.0, 0.0],
                   [0.0, -330.0, -65.0],
                   [-255.0, 170.0, -135.0],
                   [225.0, 170.0, -135.0],
                   [-150.0, -150.0, -125.0],
                   [150.0, -150.0, -125.0]]
    return np.array(modelPoints, dtype=np.float64)


def ref2DImagePoints(shape):
    '''Take list of coordinates of facial landmarks as parameterand returns the coordinates of few selected landmarks in a numpy array'''
    imagePoints = [[shape[1][0], shape[1][1]],
                   [shape[152][0], shape[152][1]],
                   [shape[33][0], shape[33][1]],
                   [shape[263][0], shape[263][1]],
                   [shape[62][0], shape[62][1]],
                   [shape[206][0], shape[206][1]]]
    return np.array(imagePoints, dtype=np.float64)

def alarm(m):
    while alarm_status:
        print("Call")
        myText = m
        language = 'en'
        output = gTTS(text=myText, lang=language, slow=False)
        output.save("output.mp3")
        playsound("output.mp3")

    if alarm_status2:
        print("Call")
        saying = True
        myText = m
        language = 'en'
        output = gTTS(text=myText, lang=language, slow=False)
        output.save("output.mp3")
        playsound("output.mp3")
        saying = False


def final_ear(shape):
    lefteye_t = [shape[157], shape[158], shape[159], shape[160], shape[161]]
    lefteye_l = [shape[154], shape[153], shape[145], shape[144], shape[163]]
    leftdist = 0
    for id in range(5):
        leftdist += math.sqrt((lefteye_l[id][0] - lefteye_t[id][0]) ** 2 + (lefteye_l[id][1] - lefteye_t[id][1]) ** 2)
    leftacross = np.linalg.norm(shape[33] - shape[133])
    earLeft = leftdist / (5 * leftacross)
    righteye_t = [shape[384], shape[385], shape[386], shape[387], shape[388]]
    righteye_l = [shape[381], shape[380], shape[374], shape[373], shape[390]]
    rightdist = 0
    for id in range(5):
        rightdist += math.sqrt((righteye_l[id][0] - righteye_t[id][0]) ** 2 + (righteye_l[id][1] - righteye_t[id][1]) ** 2)
    rightacross = np.linalg.norm(shape[362] - shape[263])
    earRight = rightdist / (5 * rightacross)

    return (earLeft + earRight) / 2


def lip_distance(shape):
    
    top_lip = shape[11:14]
    top_lip = np.concatenate((top_lip, shape[72:74], shape[81:83], shape[302:304],shape[311:313]))

    low_lip = shape[14:17]
    low_lip = np.concatenate((low_lip, shape[315:318], shape[402:405], shape[85:88], shape[178:181]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0                    
cap = cv2.VideoCapture(0)
drowsiness_time=0
yawn_time=0
looking_right_time=0
looking_left_time=0
forward_time=0
face_not_found_time=0
total_time=0
mtx,dist=cam_callibration()
#customisation part
choice=input("Do you want to receive drowsiness alert to keep you awake? Press Y for yes and N for no: ")
choice=choice.upper()
if choice=='Y':
    user_time=input("After how much time do you want to receive update(in sec): ")
while True:
    start_time=time.time()
    total_time+=time.time()-start_time
    
    GAZE = "Face Not Found"
    ret, img = cap.read()

   

    if ret:
        gray = cv2.cvtColor(img, cv2.cv2.COLOR_BGR2GRAY)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)
        if results.detections:
            for id, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = [int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)]
                im = imgRGB[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
            #            cv2.imshow("indivdual",im)
            shape = faceMesh.process(imgRGB)

            if shape.multi_face_landmarks:
                for faceLms in shape.multi_face_landmarks:
                    mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)
                    land = []
                    for id, lm in enumerate(faceLms.landmark):
                        # print(lm)
                        ih, iw, ic = img.shape
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        land.append([x, y])
                    land = np.array(land)
                
                    ear = final_ear(land)


                    if ear < EYE_AR_THRESH:
                        print(COUNTER)
                        COUNTER += 1
                        

                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            drowsiness_time+=time.time()-start_time
                            print(drowsiness_time)
                            
                            if not alarm_status and choice=="Y" and int(drowsiness_time)>int(user_time):
                                alarm_status = True
                                t = Thread(target=alarm, args=('wake up',))
                                t.deamon = True
                                t.start()

                            cv2.putText(img, "Eye closed!", (150, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 20, 255), 2)

                    else:
                        COUNTER = 0
                        alarm_status = False
                    face3Dmodel = ref3DModel()
                    distance = lip_distance(land)
                    #distance=20

                    refImgPts = ref2DImagePoints(land.tolist())
                    height, width, channels = img.shape
                    
                    mdists = np.zeros((4, 1), dtype=np.float64)

                    success, rotationVector, translationVector = cv2.solvePnP(face3Dmodel, refImgPts, mtx, dist)

                    noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
                    noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector, mtx,
                                                                mdists)

                    rmat, jac = cv2.Rodrigues(rotationVector)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    if angles[1] < -60:
                        GAZE = "Looking: Left"
                        looking_left_time+=time.time()-start_time
                    elif angles[1] > 60:
                        GAZE = "Looking: Right"
                        looking_right_time+=time.time()-start_time
                    else:
                        GAZE = "Forward"
                        forward_time+=time.time()-start_time
                    if distance > YAWN_THRESH:
                        yawn_time+=time.time()-start_time
                        cv2.putText(img, "Yawn Alert", (180, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        alarm_status2 = True
                    else:
                        alarm_status2 = False
    #Gesture control part
     # Find Hand
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=True)
        if len(lmList) != 0:

            # Filter based on size
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
            # print(area)
            if 250 < area < 1000:

                # Find Distance between index and Thumb
                length, img, lineInfo = detector.findDistance(4, 8, img)
                # print(length)


                # Convert Volume
                volBar = np.interp(length, [20, 150], [400, 150])
                volPer = np.interp(length, [20, 150], [0, 100])

                # Reduce Resolution to make it smoother
                smoothness = 10
                volPer = smoothness * round(volPer / smoothness)

                # Check fingers up
                fingers = detector.fingersUp()
                # print(fingers)

                # If pinky is down set volume
                if not fingers[3]:
                    print("Volume Changed")
                    volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    colorVol = (0, 255, 0)
                else:
                    colorVol = (255, 0, 255)

                # Drawings
                cv2.rectangle(img, (50, 150), (85, 400), (255, 255, 255), 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 255, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                            1, (255, 0, 0), 3)
                cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
                cv2.putText(img, f'Current_volume: {int(cVol)}', (250, 100), cv2.FONT_HERSHEY_COMPLEX,
                            1, colorVol, 3)

        face_not_found_time+=(total_time-looking_right_time-looking_left_time-forward_time)
        cv2.putText(img, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        cv2.putText(img,"Eye_closed_time="+str(round(drowsiness_time,2))+"   yawn_time="+str(round(yawn_time,2)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        cv2.imshow("Head Pose", img)

        key = cv2.waitKey(10) & 0xFF
        if key == 'q':
            break
    else:
        print("Failed to load webcam")
        break
cap.release()
cv2.destroyAllWindows()
