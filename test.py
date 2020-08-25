import rotate
import blurDetect as bd
import cv2
from query import q
import glob
import numpy as np
import uuid
from os import path
import util
import math


# 旋转图片
class ImRotater:
    def __init__(self):
        pass

    def rotate(self, src, angle):
        # 获取图像的尺寸
        # 旋转中心
        img = cv2.imread(src)
        (h, w) = img.shape[:2]
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

        return cv2.warpAffine(img, M, (nW, nH))


def rotate_images(srcPath, angle):
    img_files = glob.glob(srcPath + '/*.png')
    r = ImRotater()

    dst_path = "./" + str(uuid.uuid4())
    util.mkdir(dst_path)
    count = 0
    for i in img_files:

        try:
            _, ext = path.splitext(i)
            img_r = r.rotate(i, angle)
            cv2.imwrite(dst_path + "/" + str(uuid.uuid4()) + ext, img_r)
            count += 1
        except Exception as e:
            print(repr(e))
            continue
    print(f'处理图片个数:{count}')
    return dst_path


# 剪裁图片 百分比 r1,r2,c1,c2
def clip_image(src, param):
    img = cv2.imread(src)
    o = [math.ceil(img.shape[math.floor(i / 2)] * param[i]) for i in range(0, 4)]

    return img[o[0]:o[1], o[2]:o[3]]


def clip_images(srcPath, param):
    img_files = glob.glob(srcPath + '/*.png')
    r = ImRotater()

    dst_path = "./" + str(uuid.uuid4())
    util.mkdir(dst_path)
    count = 0
    for i in img_files:

        try:
            _, ext = path.splitext(i)
            img_r = clip_image(i, param)
            cv2.imwrite(dst_path + "/" + str(uuid.uuid4()) + ext, img_r)
            count += 1
        except Exception as e:
            print(repr(e))
            continue
    print(f'处理图片个数:{count}')
    return dst_path


# 镜像图片 0水平 1垂直 -1中心
def flip_image(src, param):
    img = cv2.imread(src)
    return cv2.flip(img, param)


def flip_images(srcPath, param):
    img_files = glob.glob(srcPath + '/*.png')
    r = ImRotater()

    dst_path = "./" + str(uuid.uuid4())
    util.mkdir(dst_path)
    count = 0
    for i in img_files:

        try:
            _, ext = path.splitext(i)
            img_r = flip_image(i, param)
            cv2.imwrite(dst_path + "/" + str(uuid.uuid4()) + ext, img_r)
            count += 1
        except Exception as e:
            print(repr(e))
            continue
    print(f'处理图片个数:{count}')
    return dst_path


# 处理模糊图片
def get_image_var(img_path):
    image = cv2.imread(img_path)
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img2grays = cv2.GaussianBlur(img2gray, (5, 5), 0)
    image_var = cv2.Laplacian(img2grays, cv2.CV_64F, ksize=5).var()

    # img = cv2.Laplacian(img2gray, cv2.CV_64F, ksize=5)
    return image_var / 10, image


def del_blur_images(src_path, param):
    img_files = glob.glob(src_path + '/*.png')

    dst_path = "./" + str(uuid.uuid4())
    util.mkdir(dst_path)
    count = 0
    for i in img_files:

        try:
            _, ext = path.splitext(i)
            img_var, img = get_image_var(i)
            if img_var >= param:
                cv2.imwrite(dst_path + "/" + str(uuid.uuid4()) + ext, img)
                count += 1
            else:
                continue
        except Exception as e:
            print(repr(e))
            continue
    print(f'处理图片个数:{count}')
    return dst_path


# 处理近似图片


if __name__ == '__main__':
    # dstP = rotate_images(r'E:\test\pythonCode\query', 90)
    # zipF = util.zip_dir(dstP)

    # dstP = del_blur_images(r'E:\test\pythonCode\blur', 10000)
    # zipF = util.zip_dir(dstP)

    # dstP = flip_images(r'E:\test\pythonCode\query', 1)
    # zipF = util.zip_dir(dstP)

    dstP = clip_images(r'E:\test\pythonCode\query', [0.25, 0.75, 0.2, 0.8])
    zipF = util.zip_dir(dstP)
