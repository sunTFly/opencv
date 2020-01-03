import cv2
import numpy as np
import matplotlib.pyplot as plt


# 图片的显示及灰度转换
def img():
    qq = cv2.imread("./img/a8.png")
    qqgre = cv2.imread("./img/a8.png", cv2.IMREAD_GRAYSCALE)
    b, g, r = cv2.split(qq)
    qq2 = cv2.merge((r, g, b))
    cv2.imshow("qq2", qq2)
    cutqq = qq.copy()
    cutqq[:, :, 0] = 0  # 表示b全部设置为0 下同，表示g为0
    cutqq[:, :, 1] = 0
    cv2.imshow("qq", cutqq)
    # cv2.imwrite("./img/greeqq.png",qq)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def shouimg(name, imgall):
    cv2.imshow(name, imgall)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 视频灰度转换
def video():
    v = cv2.VideoCapture("./video/xk.mp4")
    if v.isOpened():  # 判断视频是否能打开
        open, frame = v.read()
    else:
        open = False
    while open:
        ret, frame = v.read()
        if frame is None:
            break
        if ret == True:
            gry = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("vtest", frame)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    v.release()
    cv2.destroyAllWindows()


# 边界填充
def borderFill():
    qq = cv2.imread("./img/a8.png")
    b, g, r = cv2.split(qq)
    qq = cv2.merge((r, g, b))
    q1 = cv2.copyMakeBorder(qq, 50, 50, 50, 50, cv2.BORDER_REFLECT)
    q2 = cv2.copyMakeBorder(qq, 50, 50, 50, 50, cv2.BORDER_DEFAULT)
    q3 = cv2.copyMakeBorder(qq, 50, 50, 50, 50, cv2.BORDER_REFLECT_101)
    plt.subplot(2, 2, 1)
    plt.imshow(q1)
    plt.subplot(2, 2, 2)
    plt.imshow(q2)
    plt.subplot(2, 2, 3)
    plt.imshow(q3)
    plt.subplot(2, 2, 4)
    plt.imshow(qq)
    plt.show()


# 图像数值计算 addWeighted(a图，a,b图，b,c) 计算公式 a图*a+b图*b+c
def NumCalculations():
    qq = cv2.imread("./img/a8.png")
    xl = cv2.imread("./img/xl.png")
    cv2.imshow("qq", qq)
    qq1 = qq + 10
    cv2.imshow("qq1", qq1)
    qq2 = cv2.add(qq, 100)
    cv2.imshow("qq2", qq2)
    print(qq.shape)
    xl = cv2.resize(xl, (147, 129))
    he = cv2.addWeighted(qq, 0.4, xl, 0.6, 0)
    cv2.imshow("he", he)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 阈值处理 threshold参数 原图，阈值，最大值 type type确定取哪一个阈值
def detailpic():
    qq = cv2.imread("./img/a8.png")
    ret, sh1 = cv2.threshold(qq, 127, 255, cv2.THRESH_TRUNC)
    cv2.imshow('a', sh1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 图像滤波处理
def filter():
    img = cv2.imread("./img/jygree.jpg")
    img1 = cv2.blur(img, (3, 3))  # 均值滤波
    img2 = cv2.boxFilter(img, -1, (3, 3), normalize=True)  # 方框滤波
    img3 = cv2.GaussianBlur(img, (5, 5), 1)  # 高斯滤波
    img4 = cv2.medianBlur(img, 5)  # 中值滤波
    imgall = np.hstack((img, img1, img2, img3, img4))
    cv2.imshow('a', imgall)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 形态学操作
def morphology():
    img = cv2.imread('./img/fg.png')
    kernal = np.ones((5, 5), np.uint8)  # 一个卷积盒 np.uint8取 0到255整数
    imgerode = cv2.erode(img, kernal, iterations=1)  # 腐蚀操作
    imgdital = cv2.dilate(img, kernal, iterations=1)  # 膨胀操作
    openimg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernal)  # 开运算 先腐蚀，再膨胀
    colsimg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernal)  # 闭运算
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernal)  # 梯度运算 膨胀-腐蚀
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernal)  # 顶帽 原始输入-开运算
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernal)  # 黑帽 闭运算-原始输入
    imgall = np.hstack((img, colsimg, openimg))
    cv2.namedWindow("a", 0);
    cv2.resizeWindow("a", 1200, 400)


# 图像的梯度计算 cv2.Sobel(图像,cv2.CV_64F,水平方向，数值方向,像素点大小)
def grad():
    a = cv2.imread("./img/AA.png", cv2.IMREAD_GRAYSCALE)
    a = cv2.medianBlur(a, 5)
    shouimg("a", a)
    b = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=3)
    b = cv2.convertScaleAbs(b)
    shouimg("b", b)
    c = cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=3)
    c = cv2.convertScaleAbs(b)
    shouimg("c", c)
    d = cv2.addWeighted(b, 0.5, c, 0.5, 0)
    shouimg("d", d)


# 边缘检测 做的事情： ./img/byjc.png  里面
def edge():
    img = cv2.imread("./img/a8.png")
    img1 = cv2.Canny(img, 50, 100)
    img2 = cv2.Canny(img, 80, 150)
    img3 = cv2.Canny(img, 30, 90)
    imgall = np.hstack((img1, img2, img3))
    img4 = cv2.threshold()
    shouimg("f", imgall)


# 图像金字塔
def pyramid():
    img = cv2.imread('./img/xj.jpg')
    imgUp = cv2.pyrUp(img)
    imgDown = cv2.pyrDown(img)
    shouimg('img', img)
    shouimg('up', imgUp)
    shouimg('down', imgDown)
    # 拉普拉斯金字塔 原始图像减去一次下采样一次上采样的结果
    down = cv2.pyrDown(img)
    down_up = cv2.pyrUp(down)
    l = img - down_up
    shouimg('l', l)


# 模板匹配
def matchTemplate():
    img = cv2.imread('./img/cjml.jpg', 0)
    face = cv2.imread('./img/jb1.png', 0)
    imgbgr = cv2.imread('./img/cjml.jpg')
    w, h = face.shape[0:2]
    # 参数说明：模板匹配.png
    res = cv2.matchTemplate(img, face, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    loc = np.where(res >= 0.5)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(imgbgr, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
    shouimg('a', imgbgr)


matchTemplate()
