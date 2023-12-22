import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle
import time

class FaceDetection:
    def __init__(self, path_to_faceData, path_to_encodeFile):
        self.path_to_faceData = path_to_faceData
        self.path_to_encodeFile = path_to_encodeFile
        self.encodeListKnow = []
        self.classNames = []
    
    def encode(self, images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList
    
    def KhoiTao(self):
        images = []
        path = self.path_to_faceData
        myList = os.listdir(path)
        for cl in myList:
            curImg = cv2.imread(f"{path}/{cl}")
            images.append(curImg)
            self.classNames.append(os.path.splitext(cl)[0])

        # Lưu danh sách mã hóa vào file
        with open(self.path_to_encodeFile, 'wb') as f:
            pickle.dump(self.encode(images), f)
        # # Nạp danh sách mã hóa từ file
        with open(self.path_to_encodeFile, 'rb') as f:
            self.encodeListKnow = pickle.load(f)
        print("Success init!")
    
    def detect(self, frame):
        names = []
        
        # frame = face_recognition.load_image_file(path_to_image)
        framS = cv2.resize(frame, (0, 0), None, fx=0.5, fy=0.5)
        framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)

        # xac dinh vi tri khuon mat tren cam va encode tren cam
        faceCurFrame = face_recognition.face_locations(framS) # lay tung khuon mat va vi tri khuon mat hien tai
        encodeCurFrame = face_recognition.face_encodings(framS)

        for encodeface, faceLoc in zip(encodeCurFrame, faceCurFrame): # lay tung khuon mat va vi tri khuon mat hien tai theo cap
            faceDis = face_recognition.face_distance(self.encodeListKnow, encodeface)
            matchIndex = np.argmin(faceDis) # day ve index cua faceDis nho nhat

            if faceDis[matchIndex] < 0.5:
                name = self.classNames[matchIndex]
            else :
                name = "Unknow"
            names.append(name)
        
        return names

