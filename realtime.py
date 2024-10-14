import cv2
import numpy as np
from dataset import load_data

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

(X_train, y_train), (X_test, y_test) = load_data(face_cascade)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(X_train, np.array(y_train))

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_region = gray[y:y + h, x:x + w]

        label, confidence = model.predict(face_region)

        cv2.rectangle(frame, (x, y), (x + w, y +h), (255, 0, 0), 2)

        cv2.putText(frame, f'ID: {label}, Confidence: {confidence:.2f}', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Webcame - Reconnaissance faciale', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()