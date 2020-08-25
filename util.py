import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import stat
import shutil


def zip_dir(srcDir):
    dstF = srcDir + '.zip'  # 压缩后文件夹的名字
    with zipfile.ZipFile(dstF, 'w', zipfile.ZIP_DEFLATED)as z:  # 参数一：文件夹名
        imgList = glob.glob(srcDir + '/*')
        for i in imgList:
            z.write(i)

        print(f'压缩成功,生成文件:{dstF}')
        del_dir(srcDir)
        print(f'删除原文件夹成功')
    return dstF


def mkdir(srcDir):
    # 引入模块
    # 去除首位空格
    srcDir = srcDir.strip()
    # 去除尾部 \ 符号
    srcDir = srcDir.rstrip("\\")

    # 判断路径是否存在
    if not os.path.exists(srcDir):
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(srcDir)
        print(srcDir + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(srcDir + ' 目录已存在')
        return False


def del_dir(srcDir):
    if os.path.exists(srcDir):
        for fileList in os.walk(srcDir):
            for name in fileList[2]:
                os.chmod(os.path.join(fileList[0], name), stat.S_IWRITE)
                os.remove(os.path.join(fileList[0], name))
        shutil.rmtree(srcDir)
        # print(srcDir + ' 删除成功')
        return
    else:
        print(srcDir + ' 目录不存在')
        return


class ImgViewer():

    def __init__(self, img):
        # self.n = name
        self.i = img
        pass

    def cv_show(self, name):
        """图像显示函数
        name：字符串，窗口名称
        img：numpy.ndarray，图像
        """
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, self.i.shape[1], self.i.shape[0]);
        cv2.imshow(name, self.i)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def img_show(self, name):
        """matplotlib图像显示函数
        name：字符串，图像标题
        img：numpy.ndarray，图像
        """
        if len(self.i.shape) == 3:
            img = cv2.cvtColor(self.i, cv2.COLOR_BGR2RGB)
            plt.imshow(img, 'gray')
            # plt.xticks([])
            # plt.yticks([])
            plt.xlabel(name, fontproperties='FangSong', fontsize=12)
