import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def gaussKernel(sig, m=0):
    # 若m未给出，通过计算得到
    if m == 0:
        m = (math.ceil(3 * sig)) * 2 + 1
    # 若给出的m过小，给出警告提示信息
    elif m < math.ceil(3 * sig) * 2 + 1:
        print("给出的m过小")
        return

        # w = np.zeros((m, m))
    ##中心
    # mid = (m - 1)/2 + 1
    # for i in range(0, m):
    #    for j in range(0, m):
    #        value = -(pow(i-mid+1,2) + pow(j-mid+1,2))/(2*sig*sig)
    #        w[i][j] = np.exp(value)

    ##对w进行归一化处理
    ##得到w矩阵中所有元素的和
    # sum = np.sum(w)

    ##不显示科学计数法，完整显示数字
    # np.set_printoptions(suppress = True)

    # for i in range(0, m):
    #    for j in range(0, m):
    #        #w[i][j] = format(w[i][j]/sum, '.04f')
    #        w[i][j] = w[i][j]/sum

    # return w

    x = (np.arange(m) - m // 2) ** 2
    y = x.copy()
    w = y[:, None] + x[None, :]
    w = np.exp(-w.astype(np.float32) / (2 * sig * sig))

    # 进行归一化处理
    sum = np.sum(w)
    w /= sum

    return w


if __name__ == '__main__':
    res = gaussKernel(1, 7)
    # print(res)
