import cv2
import numpy as np
import untils


def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    if blurKsize >= 3:
        blurredsrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredsrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)


class VConvolutFilter(object):
    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        cv2.filter2D(src, -1, self._kernel, dst)


class SharpenFilter(VConvolutFilter):
    def __init__(self):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        VConvolutFilter.__init__(self, kernel)
