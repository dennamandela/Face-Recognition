# Import Library
import cv2
import os
import numpy as np
from PIL import Image

#initialize the recognizer and the face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector =  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def getImageAndLabels(path):

    #Load training images from a dataset folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]

    #capture the faces and Id from the training images
    faceSamples = []
    Ids = []

    #Put them In a List of Ids and FaceSamples  and return it
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(imageNp)
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y+h, x:x+w])
            Ids.append(Id)
    return faceSamples, Ids

#now we just have to call that function and feed the data to the recognizer to train
faces, Ids = getImageAndLabels('dataset')
recognizer.train(faces, np.array(Ids))
recognizer.save('training/trainingFace.yml')
    


