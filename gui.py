from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
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
from datetime import datetime
import json
from json import JSONEncoder
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import hand_detector_module as htm
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1059, 697)
        MainWindow.setStyleSheet("text-align:center;")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.learnbutton = QtWidgets.QPushButton(self.centralwidget)
        self.learnbutton.setGeometry(QtCore.QRect(180, 110, 93, 28))
        self.learnbutton.setObjectName("learnbutton")
        self.learnbutton.clicked.connect(self.click_learn)
        self.attendece_button = QtWidgets.QPushButton(self.centralwidget)
        self.attendece_button.setGeometry(QtCore.QRect(450, 110, 131, 31))
        self.attendece_button.setObjectName("attendece_button")
        #self.attendece_button.hide()
        self.terminate_button = QtWidgets.QPushButton(self.centralwidget)
        self.terminate_button.setGeometry(QtCore.QRect(780, 110, 93, 28))
        self.terminate_button.setObjectName("terminate_button")
        self.terminate_button.clicked.connect(self.click_trmnt)
        self.attendece_button.clicked.connect(self.makevisible)
        #self.attendece_button.clicked.connect(self.click_atndnc)
        self.imshow = QtWidgets.QRadioButton(self.centralwidget)
        self.imshow.setGeometry(QtCore.QRect(470, 200, 89, 20))
        self.imshow.setObjectName("imshow")
        self.input = QtWidgets.QTextEdit(self.centralwidget)
        self.input.setGeometry(QtCore.QRect(290, 10, 431, 64))
        self.input.setStyleSheet("font-size:22px;")
        self.input.setObjectName("input")
        self.input.setEnabled(False)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(300, 320, 441, 23))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.hide()
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(440, 270, 151, 41))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label.hide()
        self.attendece_button.clicked.connect(self.makevisible)
        self.yawnlab = QtWidgets.QLabel(self.centralwidget)
        self.yawnlab.setGeometry(QtCore.QRect(130, 500, 81, 16))
        self.yawnlab.setObjectName("yawnlab")
        self.rightlab = QtWidgets.QLabel(self.centralwidget)
        self.rightlab.setGeometry(QtCore.QRect(130, 380, 111, 21))
        self.rightlab.setObjectName("rightlab")
        self.leftlab = QtWidgets.QLabel(self.centralwidget)
        self.leftlab.setGeometry(QtCore.QRect(130, 430, 101, 21))
        self.leftlab.setObjectName("leftlab")
        self.drowsylab = QtWidgets.QLabel(self.centralwidget)
        self.drowsylab.setGeometry(QtCore.QRect(130, 540, 91, 16))
        self.drowsylab.setObjectName("drowsylab")
        self.right = QtWidgets.QTextBrowser(self.centralwidget)
        self.right.setGeometry(QtCore.QRect(260, 380, 256, 31))
        self.right.setObjectName("right")
        self.left = QtWidgets.QTextBrowser(self.centralwidget)
        self.left.setGeometry(QtCore.QRect(260, 420, 256, 31))
        self.left.setObjectName("left")
        self.yawn = QtWidgets.QTextBrowser(self.centralwidget)
        self.yawn.setGeometry(QtCore.QRect(240, 490, 256, 31))
        self.yawn.setObjectName("yawn")
        self.drowsy = QtWidgets.QTextBrowser(self.centralwidget)
        self.drowsy.setGeometry(QtCore.QRect(240, 530, 256, 31))
        self.drowsy.setObjectName("drowsy")
        self.volumewheel = QtWidgets.QDial(self.centralwidget)
        self.volumewheel.setGeometry(QtCore.QRect(760, 420, 111, 111))
        self.volumewheel.setAcceptDrops(False)
        self.volumewheel.setMaximum(100)
        self.volumewheel.setInvertedAppearance(False)
        self.volumewheel.setInvertedControls(False)
        self.volumewheel.setObjectName("volumewheel")
        self.volval = QtWidgets.QLabel(self.centralwidget)
        self.volval.setGeometry(QtCore.QRect(790, 460, 51, 21))
        self.volval.setText("")
        self.volval.setAlignment(QtCore.Qt.AlignCenter)
        self.volval.setObjectName("volval")
        self.volume_2 = QtWidgets.QLabel(self.centralwidget)
        self.volume_2.setGeometry(QtCore.QRect(780, 390, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.volume_2.setFont(font)
        self.volume_2.setStyleSheet("")
        self.volume_2.setObjectName("volume_2")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(740, 30, 121, 20))
        self.checkBox.setObjectName("checkBox")
        self.checkBox.clicked.connect(self.clear_prompt)
        self.a_a_f_name = QtWidgets.QComboBox(self.centralwidget)
        self.a_a_f_name.setGeometry(QtCore.QRect(740, 50, 201, 22))
        self.a_a_f_name.setObjectName("a_a_f_name")
        j = 0
        filelist = []
        for i in glob.glob('*.csv'):
            j += 1
            filelist.append(i)
        if j == 0:
            self.a_a_f_name.hide()
        else:
            bslash=chr(92)
            for i in filelist:
                name=(i.split(bslash)[-1]).split('.')[0]
                self.a_a_f_name.addItem(name)

        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(950, 50, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.click_button)
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1059, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.learnbutton.setText(_translate("MainWindow", "Learn"))
        self.attendece_button.setText(_translate("MainWindow", "take attendence"))
        self.terminate_button.setText(_translate("MainWindow", "terminate"))
        self.imshow.setText(_translate("MainWindow", "show live"))
        self.input.setHtml(_translate("MainWindow",
                                      "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                      "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                      "p, li { white-space: pre-wrap; }\n"
                                      "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:14pt; font-weight:400; font-style:normal;\">\n"
                                      "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt; color:#b0b0b0;\">Enter the name of attendence file</span></p></body></html>"))
        self.label.setText(_translate("MainWindow", "Learning........"))
        self.yawnlab.setText(_translate("MainWindow", "Yawn Time"))
        self.rightlab.setText(_translate("MainWindow", "Time seeing right"))
        self.leftlab.setText(_translate("MainWindow", "Time seeing left"))
        self.drowsylab.setText(_translate("MainWindow", "Drowsy time"))
        self.volume_2.setText(_translate("MainWindow", "volume"))
        self.checkBox.setText(_translate("MainWindow", "Edit File Name"))
        self.pushButton.setText(_translate("MainWindow", "use file"))
        #self.pushButton.setText(_translate("MainWindow", "save"))
    def click_button(self):
        fname=self.a_a_f_name.currentText()
        self.input.setText(fname)
    def show_popup(self):
        msg=QMessageBox()
        msg.setWindowTitle('File name error')
        msg.setText("Please enter a valid file name")
        msg.setIcon(QMessageBox.Critical)
        x=msg.exec_()
    def clear_prompt(self):
        if self.checkBox.isChecked()==True:
            self.input.setEnabled(True)
            if self.input.toPlainText()=='Enter the name of attendence file':
                self.input.setText('')
        else:
            if self.input.toPlainText()=='':
                self.input.setText('Enter the name of attendence file')
            self.input.setEnabled(False)
    def makevisible(self):
        if self.input.toPlainText() == 'Enter the name of attendence file':
            self.show_popup()
        else:
            self.click_atndnc()
    def click_learn(self):
        self.progressBar.show()
        self.label.show()
        self.progressBar.setValue(2)
        student_images = []

        class NumpyArrayEncoder(JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return JSONEncoder.default(self, obj)

        self.progressBar.setValue(10)
        try:
            # for i in glob.glob('Student_images/*.jpg'):
            #   student_images.append(i)
            student_images = glob.glob('Student_images/*.jpg')
            self.progressBar.setValue(45)
        except:
            print("Unable to read student data, please check data/Student_images folder")
        # student_images=glob.glob("data/Student_images/*.jpg")
        print(student_images)
        face_encoding = []
        student_names = []
        self.progressBar.setValue(60)
        self.label.setText('Learning............')
        if len(student_images) == 0:
            print("There are no images in Student_images folder")
            print(len(student_images))
            exit(0)
        self.progressBar.setValue(70)
        i=0
        val=70
        for image in student_images:
            i+=1
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encoding.append((face_recognition.face_encodings(img)[0]))
            bslash=chr(92)
            val += int(8 / i)
            self.progressBar.setValue(val)
            name = (image.split(bslash)[-1]).split('.')[0]  # Extract name from (data/Student_images/name.jpeg)-image
            student_names.append(name)
        self.progressBar.setValue(100)
        time.sleep(1)
        face_encoding = np.array(face_encoding)
        known_face_names = np.array(student_names)
        self.progressBar.hide()
        self.label.hide()
        print("Done learning and creating profiles")
        print(type(known_face_names))
        print(type(face_encoding))
        with open("Face_encodings", 'w') as f:
            json.dump(face_encoding, f, cls=NumpyArrayEncoder)
        with open("Known_face_names", 'w') as f:
            json.dump(known_face_names, f, cls=NumpyArrayEncoder)
        self.label.setText('Learnt')
    def click_atndnc(self):
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        # volume.GetMute()
        # volume.GetMasterVolumeLevel()
        volRange = volume.GetVolumeRange()

        minVol = volRange[0]
        maxVOl = volRange[1]
        vol = 0
        volBar = 400
        volPer = 0
        area = 0
        colorVol = (255, 0, 0)
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(min_detection_confidence=0.65)

        mpDraw = mp.solutions.drawing_utils

        no_of_hand = 0
        self.label.show()
        self.label.setText('Initiated..')
        self.progressBar.show()
        self.progressBar.setValue(5)
        attendanceList = []
        self.input.show()
        def makeAttendanceEntry(name, file_name):
            try:
                with open(file_name, 'r+') as FILE:
                    allLines = FILE.read().splitlines()
                    for line in allLines:
                        entry = line.split(',')
                        attendanceList.append(entry[0])
                    if (name not in attendanceList) and (name != "Unknown"):
                        now = datetime.now()
                        dtString = now.strftime('%d/%b/%Y, %H:%M:%S')
                        FILE.writelines(f'\n{name},{dtString}')
            except:
                print("Attendance list file doesn't exist")

        with open("Face_encodings", 'r') as f:
            known_face_encodings = json.load(f)
        with open("Known_face_names", 'r') as f:
            known_face_names = json.load(f)
        i=0
        valid=5
        for list in known_face_encodings:
            i+=1
            valid+=int(8/i)
            list = np.array(list)
            # self.label.show()
            self.label.setText('Creating Database...')
            # self.progressBar.show()
            self.progressBar.setValue(valid)
        known_face_encodings = np.array(known_face_encodings)
        file_name=self.filename()
        if (file_name[-4:] != ".csv"):
            file_name += ".csv"
        with open(file_name, 'w') as f:
            pass
        #self.label.show()
        self.label.setText('Database created')
        #self.progressBar.show()
        for i in range(10):
            valid += i
            self.progressBar.setValue(valid)
            time.sleep(0.01)
        # self.label.show()
        self.label.setText('Switching on Camera...')
        # self.progressBar.show()
        for i in range(10):
            valid += i
            self.progressBar.setValue(valid)
        cap = cv2.VideoCapture(0)
        # self.label.show()
        self.label.setText('Searching for a known face')
        # self.progressBar.show()
        self.progressBar.setValue(valid)

        while True:
            flag = False
            check, self.unknown_image = cap.read()
            if check:
                small_image = cv2.resize(self.unknown_image, (0, 0), None, 0.25, 0.25)
                small_image = cv2.cvtColor(self.unknown_image, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(small_image)
                face_encodings = face_recognition.face_encodings(small_image, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        self.name = known_face_names[best_match_index]
                        makeAttendanceEntry(self.name, file_name)
                    cv2.rectangle(self.unknown_image, (left, top), (right, bottom), (0, 255, 0), 3)

                    print(name)

                    cv2.putText(self.unknown_image, name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    if matches[best_match_index]:
                        # self.label.show()
                        self.label.setText('Match Found')
                        # self.progressBar.show()
                        self.progressBar.setValue(100)
                        self.name = known_face_names[best_match_index]
                        makeAttendanceEntry(self.name, file_name)
                        cv2.putText(self.unknown_image, "Match found, Proceeding to attention detection part",
                                    (left + 10, top - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        flag = True
                self.progressBar.hide()
                self.label.hide()
                #cv2.imshow("", self.unknown_image)
                #cv2.waitKey(1)
                if self.imshow.isChecked():
                    self.show_live()
                else:
                    self.show_live()
                    cv2.waitKey(1)
                if (flag):
                    break


            else:
                print("Failed to load webcam")
                break
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(2)
        # Facial features recognition part
        mpDraw = mp.solutions.drawing_utils
        mpFaceDetection = mp.solutions.face_detection
        faceDetection = mpFaceDetection.FaceDetection()
        mpFaceMesh = mp.solutions.face_mesh
        faceMesh = mpFaceMesh.FaceMesh()
        drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)
        detector = htm.handDetector(detectionCon=0.7, maxHands=1)

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

            for image in images:  # standard method to callibrate camera by finding chess board corner in an image
                img = cv2.imread(image)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

                if ret:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    imgpoints.append(corners2)
                    img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
                    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,
                                                                       None)
            return (mtx, dist)

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
                leftdist += math.sqrt(
                    (lefteye_l[id][0] - lefteye_t[id][0]) ** 2 + (lefteye_l[id][1] - lefteye_t[id][1]) ** 2)
            leftacross = np.linalg.norm(shape[33] - shape[133])
            earLeft = leftdist / (5 * leftacross)
            righteye_t = [shape[384], shape[385], shape[386], shape[387], shape[388]]
            righteye_l = [shape[381], shape[380], shape[374], shape[373], shape[390]]
            rightdist = 0
            for id in range(5):
                rightdist += math.sqrt(
                    (righteye_l[id][0] - righteye_t[id][0]) ** 2 + (righteye_l[id][1] - righteye_t[id][1]) ** 2)
            rightacross = np.linalg.norm(shape[362] - shape[263])
            earRight = rightdist / (5 * rightacross)

            return (earLeft + earRight) / 2

        def lip_distance(shape):

            top_lip = shape[11:14]
            top_lip = np.concatenate((top_lip, shape[72:74], shape[81:83], shape[302:304], shape[311:313]))

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
        self.drowsiness_time = 0
        self.yawn_time = 0
        self.looking_right_time = 0
        self.looking_left_time = 0
        self.forward_time = 0
        self.face_not_found_time = 0
        self.total_time = 0
        mtx, dist = cam_callibration()
        while True:
            start_time = time.time()
            self.total_time += time.time() - start_time

            GAZE = "Face Not Found"
            ret, self.img = cap.read()

            gray = cv2.cvtColor(self.img, cv2.cv2.COLOR_BGR2GRAY)
            imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            results = faceDetection.process(imgRGB)

            if results.detections:
                for id, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, ic = self.img.shape
                    bbox = [int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)]
                    im = imgRGB[bbox[1]:(bbox[1] + bbox[3]), bbox[0]:(bbox[0] + bbox[2])]
                #            cv2.imshow("indivdual",im)
                shape = faceMesh.process(imgRGB)

                if shape.multi_face_landmarks:
                    for faceLms in shape.multi_face_landmarks:
                        mpDraw.draw_landmarks(self.img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)
                        land = []
                        for id, lm in enumerate(faceLms.landmark):
                            # print(lm)
                            ih, iw, ic = self.img.shape
                            x, y = int(lm.x * iw), int(lm.y * ih)
                            land.append([x, y])
                        land = np.array(land)

                        ear = final_ear(land)

                        if ear < EYE_AR_THRESH:
                            print(COUNTER)
                            COUNTER += 1

                            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                self.drowsiness_time += time.time() - start_time
                                print(self.drowsiness_time)
                                #self.drowsy.setText(str(drowsiness_time))

                                if not alarm_status:
                                    alarm_status = True
                                    t = Thread(target=alarm, args=('wake up sid',))
                                    t.deamon = True
                                    t.start()

                                cv2.putText(self.img, "DROWSINESS ALERT!" + str(self.drowsiness_time), (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        else:
                            COUNTER = 0
                            alarm_status = False
                        face3Dmodel = ref3DModel()
                        distance = lip_distance(land)
                        # distance=20

                        refImgPts = ref2DImagePoints(land.tolist())
                        height, width, channels = self.img.shape

                        mdists = np.zeros((4, 1), dtype=np.float64)

                        success, rotationVector, translationVector = cv2.solvePnP(face3Dmodel, refImgPts, mtx, dist)

                        noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
                        noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector,
                                                                     mtx,
                                                                     mdists)

                        rmat, jac = cv2.Rodrigues(rotationVector)
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        if angles[1] < -60:
                            GAZE = "Looking: Left"
                            self.looking_left_time += time.time() - start_time
                        elif angles[1] >60:
                            GAZE = "Looking: Right"
                            self.looking_right_time += time.time() - start_time
                        else:
                            GAZE = "Forward"
                            self.forward_time += time.time() - start_time
                        if distance > YAWN_THRESH:
                            self.yawn_time += time.time() - start_time
                            cv2.putText(self.img, "Yawn Alert", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            alarm_status2 = True
                        else:
                            alarm_status2 = False
            self.face_not_found_time += (self.total_time - self.looking_right_time - self.looking_left_time - self.forward_time)
            self.drowsy.setText(str(round(self.drowsiness_time,2))+"sec")
            self.left.setText(str(round(self.looking_left_time,2))+"sec")
            self.right.setText(str(round(self.looking_right_time,2))+"sec")
            self.yawn.setText(str(round(self.yawn_time,2))+"sec")
            cv2.putText(self.img, GAZE + " " + str(round(self.drowsiness_time, 2)) + " " + str(round(self.yawn_time, 2)), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
            cv2.putText(self.img, "drowsiness_time=" + str(round(self.drowsiness_time, 2)) + "   yawn_time=" + str(
                round(self.yawn_time, 2)), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
            rgb_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            img = detector.findHands(self.img)
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
                    if not fingers[4]:
                        volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                        cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                        colorVol = (0, 255, 0)
                    else:
                        colorVol = (255, 0, 0)

    # Drawings
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 0, 0), 3)
            cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
            cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX,
                        1, colorVol, 3)
            #key = cv2.waitKey(10) & 0xFF
            if self.imshow.isChecked():
                self.show_liv()
            else:
                self.show_liv()
                cv2.waitKey(1)
            #if key == 'q':
            #    break



        cap.release()
        cv2.destroyAllWindows()

    def click_trmnt(self):
        self.add_to_excell()
        cv2.destroyAllWindows()
        exit(0)
    def show_live(self):
        if self.imshow.isChecked():
            #self.imshow.click()
            cv2.imshow("", self.unknown_image)
            cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()
    def show_liv(self):
        if self.imshow.isChecked():
            #self.imshow.click()
            cv2.imshow("Head Pose", self.img)
            cv2.waitKey(1)
        else:
            cv2.destroyAllWindows()
    def filename(self):
        name=self.input.toPlainText()
        return name
    def add_to_excell(self):
        with open(self.filename()+'.csv', 'a') as FILE:
            FILE.writelines(f'''\nDuration of looking left-->{str(round(self.looking_left_time,2))}
Duration of looking right-->{str(round(self.looking_right_time,2))}


Yawning duration-->{str(round(self.yawn_time,2))}
Drowsy Duration-->{str(round(self.drowsiness_time,2))}''')



if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())