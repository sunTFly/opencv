import numpy as np
import cv2

'''
img——待分割的源图像，必须是8位3通道，在处理的过程中不会被修改
mask——掩码图像，如果使用掩码进行初始化，那么mask保存初始化掩码信息；在执行分割的时候，
也可以将用户交互所设定的前景与背景保存到mask中，然后再传入grabCut函数；在处理结束之后，mask中会保存结果。mask只能取以下四种值：
GCD_BGD（=0），背景；
GCD_FGD（=1），前景；
GCD_PR_BGD（=2），可能的背景；
GCD_PR_FGD（=3），可能的前景。
              如果没有手工标记GCD_BGD或者GCD_FGD，那么结果只会有GCD_PR_BGD或GCD_PR_FGD；
rect——用于限定需要进行分割的图像范围，只有该矩形窗口内的图像部分才被处理；
bgdModel——背景模型，如果为None，函数内部会自动创建一个bgdModel；bgdModel必须是单通道浮点型图像，且行数只能为1，列数只能为13x5；
fgdModel——前景模型，如果为None，函数内部会自动创建一个fgdModel；fgdModel必须是单通道浮点型图像，且行数只能为1，列数只能为13x5；
iterCount——迭代次数，必须大于0；
mode——用于指示grabCut函数进行什么操作，可选的值有：
GC_INIT_WITH_RECT（=0），用矩形窗初始化GrabCut；
GC_INIT_WITH_MASK（=1），用掩码图像初始化GrabCut；
GC_EVAL（=2），执行分割。
'''

img = cv2.imread('./image/ff.png')
# 创建一个和图像同样大小的掩模，并用0填充
mask = np.zeros(img.shape[:2], np.uint8)
# print(img.shape[:2])
# 创建以0填充的背景和前景模型
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
# rect 前景的矩形，格式为（x,y,w,h），分别为左上角坐标和宽度，高度
rect = (0, 20, 197, 282)

'''
img 输入图像
mask 蒙板图像，确定前景区域，背景区域，不确定区域，可以设置为cv2.GC_BGD,cv2.GC_FGD,cv2.GC_PR_BGD,cv2.GC_PR_FGD，也可以输入0,1,2,3
rect 前景的矩形，格式为（x,y,w,h），分别为左上角坐标和宽度，高度
bdgModel, fgdModel 算法内部是用的数组，只需要创建两个大小为(1,65）np.float64的数组。
iterCount 迭代次数
mode cv2.GC_INIT_WITH_RECT 或 cv2.GC_INIT_WITH_MASK，使用矩阵模式还是蒙板模式。
'''

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
# print(mask)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# np.newaxis 增加一维； 每个通道都进行相应的变换
img1 = img * mask2[:, :, np.newaxis]
imgall = np.hstack((img1, img))
cv2.imshow('a', imgall)
cv2.waitKey(0)
cv2.destroyAllWindows()
