import cv2


def detect():
    face_cascades = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    eye_cascades = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascades.detectMultiScale(gray, 1.3, 2)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:(y + h), x:(x + w)]
            eyes = eye_cascades.detectMultiScale(roi_gray, 1.03, 2, 0, (40, 40))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv2.imshow('c', frame)
        if cv2.waitKey(1000 // 12) & 0xff == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect()
