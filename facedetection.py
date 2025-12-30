# for opencv insatallation : pip install opencv-python

import sys
import cv2

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print(f"ERROR: Could not load cascade file at: {cascade_path}")
    print("Make sure OpenCV is installed and the file exists, or provide a correct path.")
    sys.exit(1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('ERROR: Could not open video capture (camera).')
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print('WARNING: Frame not received from camera, retrying...')
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
