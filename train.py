import face_recognition
import numpy as np
from PIL import Image, ImageDraw
import cv2
from datetime import datetime
import json
from json import JSONEncoder
import glob
import os



class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
try:
    student_images=glob.glob("data/Student_images/*.jpg")
except:
    print("Unable to read student data, please check data/Student_images folder")
#student_images=glob.glob("data/Student_images/*.jpg")
face_encoding=[]
student_names = []
if not len(student_images):
    print("There are no images in Student_images folder")
    print(len(student_images))
    exit(0)
    
for image in student_images:
    img=cv2.imread(image)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    face_encoding.append((face_recognition.face_encodings(img)[0]))
    name=(image.split('/')[-1]).split('.')[0] #Extract name from (data/Student_images/name.jpeg)-image
    student_names.append(name)



face_encoding=np.array(face_encoding)
known_face_names=np.array(student_names)

print("Done learning and creating profiles")
print(type(known_face_names))
print(type(face_encoding))
with open("Face_encodings",'w') as f:
    json.dump(face_encoding,f,cls=NumpyArrayEncoder)
with open("Known_face_names",'w') as f:
   json.dump(known_face_names,f,cls=NumpyArrayEncoder)
