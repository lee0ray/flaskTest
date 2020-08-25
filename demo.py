import os
import glob
import cv2

def getMatchNum(matches, ratio):
    '''返回特征点匹配数量和匹配掩码'''
    matchesMask = [[0, 0] for x in range(len(matches))]
    matchNum = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:  # 将距离比率小于ratio的匹配点删选出来
            matchesMask[i] = [1, 0]
            matchNum += 1
    return (matchNum, matchesMask)

    # 创建特征提取器

query_path = 'E:/test/pythonCode/query'  # 图库路径
sample_path = query_path + '/100.png'  # 样本图片
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
    (matchNum, matchesMask) = getMatchNum(matches, 0.9)  # 通过比率条件，计算出匹配程度
    matchRatio = matchNum * 100 / len(matches)
    drawParams = dict(matchColor=(0, 255, 0),
                      singlePointColor=(255, 0, 0),
                      matchesMask=matchesMask,
                      flags=0)
    comparisonImage = cv2.drawMatchesKnn(sample_image, kp1, queryImage, kp2, matches, None, **drawParams)
    comparisonImageList.append((comparisonImage, matchRatio))  # 记录下结果
    comparisonImageList.sort(key=lambda x: x[1], reverse=True)


# l = glob.glob('./demo/*')
# repeat_img = []
# for i in l:
#     try:
#         img = cv2.imread(i)
#         cv2.imshow('1', img)
#     except Exception as e:
#         print(repr(e))

for i,s in comparisonImageList:
    cv2.imshow(f"{s:.2f}",i)

cv2.waitKey(0)
cv2.destroyAllWindows()
