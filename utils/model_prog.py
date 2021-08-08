from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

                    
model = load_model('utils/training_face.model')


#Reading an image
def final_predict(img):

        image = cv2.imread(img)
        
        classes = ['male','female']

        face, confidence = cv.detect_face(image)

        for idx, f in enumerate(face):

                # get corner points of face rectangle        
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

        #Pre-Processing to predict Gender
        face_crop = np.copy(image[startY:endY, startX:endX])
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        prediction = model.predict(face_crop)[0]

        idx = np.argmax(prediction)
        label = classes[idx]
        return label
