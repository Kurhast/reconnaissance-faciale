import cv2
import os
import numpy as np

def extract_face(filepath, face_cascade, size=(200, 200)):

    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)

    if len(faces) != 1:
        return None

    (x, y, w, h) = faces[0]

    face_region = image[y:y + h, x:x + w]

    face_region = cv2.resize(face_region, size)

    return face_region

def load_data(face_cascade, data_dir='yalefaces'):

    X_train, y_train = [], []
    X_test, y_test = [], []

    for filename in os.listdir(data_dir):

        if 'test' in filename:
            test = True 
        else:
            test = False 

        filepath = os.path.join(data_dir, filename)
        face = extract_face(filepath, face_cascade)

        if face is not None:

            label = int(filename.split('.')[0].replace('subject', ''))

            if test:
                X_test.append(face)
                y_test.append(label)
            else:
                X_train.append(face)
                y_train.append(label)

    X_train = np.array(X_train, dtype='object')
    y_train = np.array(y_train, dtype=np.int32)
    X_test = np.array(X_test, dtype='object')
    y_test = np.array(y_test, dtype=np.int32)

    return (X_train, y_train), (X_test, y_test)