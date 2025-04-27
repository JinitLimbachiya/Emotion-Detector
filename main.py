#### TO USE THIS MODEL, RUN THIS COMMAND IN TERMINAL 'pip install cv2 numpy tensorflow' ####

import cv2
import numpy as np 
from tensorflow.keras.models import load_model

model = load_model('emotion_model.h5', compile=False)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv2.VideoCapture(0)

lastFace = None
frames_since_last_detection = 0
maxFrames = 5

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        lastFace = (x, y, w, h)
        frames_since_last_detection = 0

    else:
        frames_since_last_detection += 1

    if lastFace is not None and frames_since_last_detection <= maxFrames:
        (x, y, w, h) = lastFace

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)
        labelIndex = np.argmax(prediction)
        emotionText = emotions[labelIndex]

        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotionText, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()