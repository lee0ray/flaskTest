import cv2
import sys
import numpy as np


def normalization(data):
    _range = np.max(abs(data))

    try:
        result = data / _range
    except ZeroDivisionError:
        print("division by zero!")
        return data
    else:
        print("result is", result)
    finally:
        print("executing finally clause")


def standardization(data):
    # mu = np.mean(data, axis=0)
    # sigma = np.std(data, axis=0)
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


def getImageVar(imgPath):
    image = cv2.imread(imgPath)
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img2grays = cv2.GaussianBlur(img2gray, (5, 5), 0)
    imageVar = cv2.Laplacian(img2grays, cv2.CV_64F, ksize=5).var()

    img = cv2.Laplacian(img2gray, cv2.CV_64F, ksize=5)
    return imageVar, img


if __name__ == "__main__":
    img = cv2.imread("blank.PNG")
    cv2.imshow("1",img)
    # cv2.normalize()
    # cv2.norm()

    var1, img1 = getImageVar("1.jpg")
    var2, img2 = getImageVar("car1.png")
    var3, img3 = getImageVar("car2.png")
    var4, img4 = getImageVar("blank.PNG")
    # cv2.imshow("1", img1)
    # cv2.imshow("2", img2)
    # cv2.imshow("3", img3)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

