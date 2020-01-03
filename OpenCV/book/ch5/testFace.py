import cv2
import numpy as np


def generate():
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    eye_cascades = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = camera.read()
        frame = np.fliplr(frame).copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            cv2.imwrite('./image/ls/%s.pgm' % str(count), f)
            count += 1
        cv2.imshow('c', frame)
        if cv2.waitKey(1000 // 12) & 0xff == ord('q') & count == 20:
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    generate()
