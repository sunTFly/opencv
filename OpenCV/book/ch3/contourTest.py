import cv2
import numpy as np
from scipy import ndimage


'''opencv视频中 图像金字塔里有轮廓检测
'''


def imgshow(name, pic):
    cv2.imshow(name, pic)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 高通滤波 边缘提取与增强   低通滤波：边缘平滑
kernel1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
kernel2 = np.array([[-1, -1, -1, -1, -1], [-1, 1, 2, 1, -1],
                    [-1, 2, 4, 2, -1], [-1, 1, 2, 1, -1], [-1, -1, -1, -1, -1]])
img = cv2.imread('./img/ml1.jpg', 0)
k3 = ndimage.convolve(img, kernel1)
k5 = ndimage.convolve(img, kernel2)
# 高斯模糊后进行差值计算
blurred = cv2.GaussianBlur(img, (11, 11), 0)
g_hpf = img - blurred

'''
1. cv2.cvtcolor(img, cv2.COLOR_BGR2GRAY) # 将彩色图转换为灰度图

参数说明: img表示输入的图片， cv2.COLOR_BGR2GRAY表示颜色的变换形式

2. cv2.findContours(img，mode, method)  # 找出图中的轮廓值，得到的轮廓值都是嵌套格式的

参数说明:img表示输入的图片，mode表示轮廓检索模式，通常都使用RETR_TREE找出所有的轮廓值，method表示轮廓逼近方法，使用NONE表示所有轮廓都显示

3. cv2.drawCountours(img, contours, -1, (0, 0, 255), 2) # 画出图片中的轮廓值，也可以用来画轮廓的近似值

参数说明:img表示输入的需要画的图片， contours表示轮廓值，-1表示轮廓的索引，(0, 0, 255)表示颜色， 2表示线条粗细

4. cv2.contourArea(cnt， True)  # 计算轮廓的面积

参数说明：cnt为输入的单个轮廓值

5. cv2.arcLength(cnt， True)   #  计算轮廓的周长

参数说明：cnt为输入的单个轮廓值
6. cv2.aprroxPolyDP(cnt, epsilon， True)  # 用于获得轮廓的近似值，使用cv2.drawCountors进行画图操作

 参数说明：cnt为输入的轮廓值， epsilon为阈值T，通常使用轮廓的周长作为阈值，True表示的是轮廓是闭合的

7. x, y, w, h = cv2.boudingrect(cnt) # 获得外接矩形

参数说明：x，y, w, h 分别表示外接矩形的x轴和y轴的坐标，以及矩形的宽和高， cnt表示输入的轮廓值

8 cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 根据坐标在图像上画出矩形

参数说明: img表示传入的图片， (x, y)表示左上角的位置, （x+w， y+h）表示加上右下角的位置，（0, 255, 0)表示颜色，2表示线条的粗细

9. (x, y), radius = cv2.minEnclosingCircle(cnt) # 获得外接圆的位置信息

参数说明: (x, y)表示外接圆的圆心，radius表示外接圆的半径， cnt表示输入的轮廓

10. cv2.Cricle(img, center, radius, (0, 255, 0), 2)  # 根据坐标在图上画出圆

参数说明:img表示需要画的图片，center表示圆的中心点，radius表示圆的半径, (0, 255, 0)表示颜色， 2表示线条的粗细

轮廓检测：轮廓检测相较于canny边缘检测，轮廓检测的线条要更少一些，在opencv中，使用的函数是cv2.findCountor进行轮廓检测
'''


def contour():
    img3 = cv2.imread('./img/hammer.jpg', cv2.IMREAD_UNCHANGED)
    # cv2.pyrDown() 从一个高分辨率大尺寸的图像向上构建一个金字塔（尺寸变小，分辨率降低）
    img = cv2.pyrDown(img3)
    # imgAll = np.hstack((img, k3, k5, blurred, g_hpf))
    # 阈值处理 threshold参数 原图，阈值，最大值 type type确定取哪一个阈值 -----图 阈值.png
    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    '''
     cv2.findContours(img，mode, method)  # 找出图中的轮廓值，得到的轮廓值都是嵌套格式的
    
    参数说明:img表示输入的图片，mode表示轮廓检索模式，通常都使用RETR_TREE找出所有的轮廓值，
    method表示轮廓逼近方法，使用NONE表示所有轮廓都显示
    返回结果 ：第一个值是做完2值的结果 thresh，第二个值保存的轮廓信息，最后一个返回的是层级
    '''
    image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        '''
    cv2.boundingRect(img)
    img是一个二值图，也就是它的参数；
    返回四个值，分别是x，y，w，h；
    x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    然后利用cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)画出矩行
        '''
        x, y, w, h = cv2.boundingRect(c)
        '''
        参数解释
    第一个参数：img是原图
    第二个参数：（x，y）是矩阵的左上点坐标
    第三个参数：（x+w，y+h）是矩阵的右下点坐标
    第四个参数：（0,255,0）是画线对应的rgb颜色
    第五个参数：2是所画的线的宽度
        '''
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 生成最小外接矩形
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        '''
         cv2.drawCountours(img, contours, -1, (0, 0, 255), 2) # 画出图片中的轮廓值，也可以用来画轮廓的近似值
    
    参数说明:img表示输入的需要画的图片， contours表示轮廓值，-1表示轮廓的索引，(0, 0, 255)表示颜色， 2表示线条粗细
        '''
        cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        img = cv2.circle(img, center, radius, (0, 255, 0), 2)
    # cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
    (x, y), radius = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = int(radius)
    img = cv2.circle(img, center, radius, (0, 255, 0), 2)
    # imgshow('all', img)
    for cnt in contours:
        # 轮廓周长
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        # 对图像轮廓点进行多边形拟合
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # 寻找图像的凸包
        hull = cv2.convexHull(cnt)
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
        cv2.drawContours(img, [approx], -1, (255, 255, 0), 2)
        cv2.drawContours(img, [hull], -1, (0, 0, 255), 2)
    imgshow('all', img)


'''
主要有cv2.line()//画线， cv2.circle()//画圆， cv2.rectangle()//长方形，cv2.ellipse()//椭圆， cv2.putText()//文字绘制
主要参数
img：源图像
color：需要传入的颜色
thickness：线条的粗细，默认值是1
linetype：线条的类型，8 连接，抗锯齿等。默认情况是 8 连接。cv2.LINE_AA 为抗锯齿，这样看起来会非常平滑。
'''


# 直线检测
def lines():
    img = cv2.imread('./img/lines.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''
    image：源图像
    threshold1：阈值1
    threshold2：阈值2
    apertureSize：可选参数，Sobel算子的大小
    其中，较大的阈值2用于检测图像中明显的边缘，但一般情况下检测的效果不会那么完美，
    边缘检测出来是断断续续的。所以这时候用较小的第一个阈值用于将这些间断的边缘连接起来。
    函数返回的是二值图，包含检测出的边缘
    '''
    edges = cv2.Canny(gray, 50, 120)
    minLineLenght = 20
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLenght, maxLineGap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    imgshow('ed', edges)
    imgshow('all', img)


# 圆检测
def Circles():
    planets = cv2.imread('./img/planet_glow.jpg')
    gray_img = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray_img, 5)

    '''
    image:8位，单通道图像。如果使用彩色图像，需要先转换为灰度图像。
method：定义检测图像中圆的方法。目前唯一实现的方法是cv2.HOUGH_GRADIENT。
dp：累加器分辨率与图像分辨率的反比。dp获取越大，累加器数组越小。
minDist：检测到的圆的中心，（x,y）坐标之间的最小距离。如果minDist太小，则可能导致检测到多个相邻的圆。如果minDist太大，则可能导致很多圆检测不到。
param1：用于处理边缘检测的梯度值方法。
param2：cv2.HOUGH_GRADIENT方法的累加器阈值。阈值越小，检测到的圈子越多。
minRadius：半径的最小大小（以像素为单位）。
maxRadius：半径的最大大小（以像素为单位）。
    '''
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 120,
                               param1=100, param2=30, minRadius=0, maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(planets, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(planets, (i[0], i[1]), 2, (0, 0, 255), 3)
    imgshow('p', planets)


contour()
