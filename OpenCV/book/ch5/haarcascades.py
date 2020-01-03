import cv2

filename = './image/timg.jpg'


def detect(filename):
    face_cascades = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 参数：灰度图，每次迭代时图像的压缩率，每个人脸矩形保留近邻数目的最小值
    faces = face_cascades.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        '''
         参数解释
     第一个参数：img是原图
     第二个参数：（x，y）是矩阵的左上点坐标
     第三个参数：（x+w，y+h）是矩阵的右下点坐标
     第四个参数：（0,255,0）是画线对应的rgb颜色
     第五个参数：2是所画的线的宽度
         '''
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow('The Avengers', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detect(filename)
