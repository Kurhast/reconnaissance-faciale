import cv2
import argparse
import numpy as np
from dataset import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--classifier', '-c', choices=['lbp', 'eigen', 'fisher'], default='lbp')
args = parser.parse_args()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

(X_train, y_train), (X_test, y_test) = load_data(face_cascade)
print(X_train)
if args.classifier == 'lbp':
    model = cv2.face.LBPHFaceRecognizer_create()
elif args.classifier == 'eigen':
    model = cv2.face.EigenFaceRecognizer_create()
elif args.classifier == 'fisher':
    model = cv2.face.FisherFaceRecognizer_create()


model.train(X_train, np.array(y_train))

correct_predictions = 0

for i in range(len(X_test)):
    label, confidence = model.predict(X_test[i])
    if label == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(X_test)
# print(f"Taux de précision: {accuracy * 100:.2f}%")