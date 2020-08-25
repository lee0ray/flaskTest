import os
import cv2
import glob
import numpy as np
from util import ImgViewer
import matplotlib.pyplot as plt


def rotate_bound(image, angle):
    # 获取图像的尺寸
    # 旋转中心
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    # 设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像旋转后的新边界
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    return cv2.warpAffine(image, M, (nW, nH))


# rootdir = 'F:\data'
# list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
# for i in range(0, len(list)):
#     path = os.path.join(rootdir, list[i])
#     if os.path.isfile(path):
#         cv.imread()
# # 你想对文件的操作

# try:
#     1/0
# except Exception as e:
#     # 访问异常的错误编号和详细信息
#     print(e.args)
#     print(str(e))
#     print(repr(e))


def rotate():
    imgFiles = glob.glob('./*.jpg')

    for i in range(0, len(imgFiles)):
        pass
    pass


if __name__ == '__main__':
    img = cv2.imread('1.jpg')

    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    plt.subplot(121);
    plt.imshow(img)
    plt.subplot(122);
    plt.imshow(img2)
    plt.show()

    plt.imshow(img)
    plt.show()
    imgR = cv2.rotate(img, rotateCode=cv2.ROTATE_90_CLOCKWISE)


    s = ImgViewer(img)
    s2 = ImgViewer(imgR)

    # s.cv_show("origin")
    # s2.cv_show("rotated")
