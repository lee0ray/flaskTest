import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import math
import glob
import time


def getMatchNum(matches, ratio):
    '''返回特征点匹配数量和匹配掩码'''
    matchesMask = [[0, 0] for x in range(len(matches))]
    matchNum = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:  # 将距离比率小于ratio的匹配点删选出来
            matchesMask[i] = [1, 0]
            matchNum += 1
    return (matchNum, matchesMask)


def sift_compare(sample_path, query_path):
    # 创建SIFT特征提取器
    comparisonImageList = []  # 记录比较结果
    sift = cv2.SIFT_create()
    # 创建FLANN匹配对象
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    sample_image = cv2.imread(sample_path, 0)
    kp1, des1 = sift.detectAndCompute(sample_image, None)  # 提取样本图片的特征
    files = glob.glob(query_path + '/*.png')

    for p in files:

        queryImage = cv2.imread(p, 0)
        kp2, des2 = sift.detectAndCompute(queryImage, None)  # 提取比对图片的特征
        matches = flann.knnMatch(des1, des2, k=2)  # 匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
        (matchNum, matchesMask) = getMatchNum(matches, 0.85)  # 通过比率条件，计算出匹配程度
        matchRatio = matchNum * 100 / len(matches)
        drawParams = dict(matchColor=(0, 255, 0),
                          singlePointColor=(255, 0, 0),
                          matchesMask=matchesMask,
                          flags=0)
        comparisonImage = cv2.drawMatchesKnn(sample_image, kp1, queryImage, kp2, matches, None, **drawParams)
        comparisonImageList.append((comparisonImage, matchRatio))  # 记录下结果
        comparisonImageList.sort(key=lambda x: x[1], reverse=True)
    return comparisonImageList  # 按照匹配度排序


def orb_compare(sample_path, query_path):
    # 创建特征提取器
    comparisonImageList = []  # 记录比较结果
    orb = cv2.ORB_create()
    # 创建FLANN匹配对象
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, searchParams)

    sample_image = cv2.imread(sample_path, 0)
    kp1, des1 = orb.detectAndCompute(sample_image, None)  # 提取样本图片的特征
    files = glob.glob(query_path + '/*.png')

    for p in files:
        queryImage = cv2.imread(p, 0)
        kp2, des2 = orb.detectAndCompute(queryImage, None)  # 提取比对图片的特征
        matches = flann.knnMatch(des1, des2, k=2)  # 匹配特征点，为了删选匹配点，指定k为2，这样对样本图的每个特征点，返回两个匹配
        (matchNum, matchesMask) = getMatchNum(matches, 0.8)  # 通过比率条件，计算出匹配程度
        matchRatio = matchNum * 100 / len(matches)
        drawParams = dict(matchColor=(0, 255, 0),
                          singlePointColor=(255, 0, 0),
                          matchesMask=matchesMask,
                          flags=0)
        comparisonImage = cv2.drawMatchesKnn(sample_image, kp1, queryImage, kp2, matches, None, **drawParams)
        comparisonImageList.append((comparisonImage, matchRatio))  # 记录下结果
        comparisonImageList.sort(key=lambda x: x[1], reverse=True)
    return comparisonImageList  # 按照匹配度排序


# count = len(comparisonImageList)
# column = 1
# row = math.ceil(count / column)
# 绘图显示
# figure, ax = plt.subplots(row, column)
# for index, (image, ratio) in enumerate(comparisonImageList):
#     ax[int(index / column)].set_title('Similiarity %.2f%%' % ratio)
#     ax[int(index / column)].imshow(image)
# plt.show()

if __name__ == '__main__':
    path = 'E:/test/pythonCode/query'  # 图库路径
    samplePath = path + '/100.png'  # 样本图片

    t1 = time.time()
    l1 = sift_compare(samplePath, path)
    t2 = time.time()
    sift_t = t2 - t1

    t3 = time.time()
    l2 = orb_compare(samplePath, path)
    t4 = time.time()
    orb_t = t4 - t3

    for (im, score) in l1:
        cv2.imshow(f"sift {score:.2f}", im)

    for (im, score) in l2:
        cv2.imshow(f"orb {score:.2f}", im)

    print(f'sift:{sift_t:.2f}s,orb:{orb_t:.2f}s')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
