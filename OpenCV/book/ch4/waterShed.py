import numpy as np
import cv2


'''
1. 将白色背景编程黑色背景 - 目的是为了后面变的变换做准备
2. 使用filter2D与拉普拉斯算子实现图像对比度的提高 - sharp
3. 转为二值图像通过threshold
4. 距离变换
5. 对距离变换结果进行归一化[0-1]之间
6. 使用阈值，在此二值化，得到标记
7. 腐蚀每个peak erode
8. 发现轮廓 findContours
9. 绘制轮廓 drawContours
10.分水岭变换 watershed
11.对每个分割区域着色输出结果

'''

img = cv2.imread('./image/basil.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 阈值处理 threshold参数 原图，阈值，最大值 type type确定取哪一个阈值
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# 进行开运算（先腐蚀，再膨胀） 去除噪音点
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
# 进行膨胀
sure_bg = cv2.dilate(opening, kernel, iterations=3)
'''
计算图像中每一个非零点距离离自己最近的零点的距离，
distanceTransform的第二个Mat矩阵参数dst保存了每一个点与最近的零点的距离信息，图像上越亮的点，代表了离零点的距离越远。
src – 8-bit, 单通道（二值化）输入图片。
dst – 输出结果中包含计算的距离，这是一个32-bit  float 单通道的Mat类型数组，大小与输入图片相同。
src – 8-bit, 单通道（二值化）输入图片。
dst – 输出结果中包含计算的距离，这是一个32-bit  float 单通道的Mat类型数组，大小与输入图片相同。
distanceType – 计算距离的类型那个，可以是 CV_DIST_L1、CV_DIST_L2 、CV_DIST_C。
maskSize – 距离变换掩码矩阵的大小，可以是
3（CV_DIST_L1、 CV_DIST_L2 、CV_DIST_C）
5（CV_DIST_L2 ）
CV_DIST_MASK_PRECISE (这个只能在4参数的API中使用)
labels – 可选的2D标签输出（离散 Voronoi 图），类型为 CV_32SC1 大小同输入图片。
labelType – 输出标签的类型，这里有些两种。
labelType==DIST_LABEL_CCOMP 将周围较近的白色像素点作为一个整体计算其到黑色边缘的距离
labelType==DIST_LABEL_PIXEL 单独计算每个白色像素点到其黑色边缘的距离.
'''
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
cv2.imshow('c', sure_bg)
cv2.imshow('b', sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
cv2.imshow('d', unknown)
# 过滤原始图像中轮廓分析后较小的区域，留下较大区域。
ret, markers = cv2.connectedComponents(sure_fg)
# 将标记加一，再将不确定区域变为0
markers = markers + 1
markers[unknown == 255] = 0
# 根据分水岭算法 ；第一个参数是原图，1为前景区域，根据原图计算前景相连节点属于背景区域还是前景区域，用红线分割
markers = cv2.watershed(img, markers)
# print(markers)
img[markers == -1] = [0, 0, 255]
cv2.imshow('a', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
