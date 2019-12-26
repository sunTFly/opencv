import cv2
import numpy as np
import sys


def test1():
    # 打开摄像头cs
    c = cv2.VideoCapture(0)
    # cv2.namedWindow('window')
    success, f = c.read()
    size = (int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # 设置写入格式 大小
    w = cv2.VideoWriter('a.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), 30, (640, 480))
    # 显示在窗口
    i = 0
    while success and cv2.waitKey(1) == -1:
        # 镜像
        f = np.fliplr(f).copy()
        # 写入
        w.write(f)
        cv2.imshow('window', f)
        success, f = c.read()
        cv2.waitKey(1)
        # 截图
        if i == 5:
            cv2.imwrite('a.png', f)
        i = i + 1

    cv2.destroyAllWindows()
    c.release()
    sys.exit()



