import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


# padding函数
def padding(f, w, method='zero'):
    # 先根据w的大小来确定f padding后的大小
    row, col = w.shape
    rowf, colf = f.shape

    # w必须是奇数的方阵
    if (row == col) and (row % 2 == 1):
        # 要填充的列数和行数
        k = math.floor(row / 2)
        # 填充后的f的行数和列数
        rowFinal = rowf + 2 * k
        colFinal = colf + 2 * k

        # 创建一个指定维度的数组，并将其填充为0
        p = np.zeros((rowFinal, colFinal))

        # 以replicate方式进行填充
        if method == 'replicate':
            # 填中间部分：赋值为原数组对应点处的像素值
            # i的下标范围为：[k, k+rowf-1]，但range范围的上限取不到
            # for i in range(k, k + rowf):
            #    for j in range(k, k + colf):
            #        p[i][j] = f[i-k][j-k]

            p[k:k + rowf, k:k + colf] = f[:, :]

            # 填最左最右列两大部分
            p[k:k + rowf, 0:k] = p[k:k + rowf, k:k + 1]
            p[k:k + rowf, k + colf:colFinal] = p[k:k + rowf, k + colf - 1:k + colf]
            # for i in range(k, k + rowf):
            #    for j in range(0, k):
            #        p[i][j] = p[i][k]
            #        #p[i][j] = f[i-k-1][0]
            #    for j in range(k + colf, colFinal):
            #        p[i][j] = p[i][k+colf-1]
            #        #p[i][j] = f[i-k-1][colf-1]

            # 填最上最下行两大部分
            p[0:k, k:k + colf] = p[k:k + 1, k:k + colf]
            p[k + rowf:rowFinal, k:k + colf] = p[k + rowf - 1:k + rowf, k:k + colf]
            # for j in range(k, k + colf):
            #    for i in range(0, k):
            #        p[i][j] = p[k][j]
            #        #p[i][j] = f[0][j-k-1]
            #    for i in range(k + rowf, rowFinal):
            #        p[i][j] = p[k+rowf-1][j]
            #        #p[i][j] = f[rowf-1][j-k-1]

            # 填四个角
            # 先填上面两个角
            p[0:k, 0:k] = p[k:k + 1, k:k + 1]
            p[0:k, k + colf:colFinal] = p[k:k + 1, k + colf - 1:k + colf]
            # for i in range(0, k):
            #    for j in range(0, k):
            #        p[i][j] = p[k][k]
            #        #p[i][j] = f[0][0]
            #    for j in range(k + colf, colFinal):
            #        p[i][j] = p[k][k+colf-1]
            #        #p[i][j] = f[0][colf-1]
            # 再填下面两个角
            p[k + rowf:rowFinal, 0:k] = p[k + rowf - 1:k + rowf, k:k + 1]
            p[k + rowf:rowFinal, k + colf:colFinal] = p[k + rowf - 1:k + rowf, k + colf - 1:k + colf]
            # for i in range(k + rowf, rowFinal):
            #    for j in range(0, k):
            #        p[i][j] = p[k+rowf-1][k]
            #        #p[i][j] = f[rowf-1][0]
            #    for j in range(k + colf, colFinal):
            #        p[i][j] = p[k+rowf-1][k+colf-1]
            #        #p[i][j] = f[rowf-1][col-1]

        # 以zero方式进行填充
        if method == 'zero':
            # 只需填充中间的部分，其余部分已填充为0
            p[k:k + rowf, k:k + colf] = f[:, :]
            # for i in range(k, k + rowf):
            #    for j in range(k, k + colf):
            #        p[i][j] = f[i-k][j-k]
    else:
        print("卷积核有误")

    return p


# 二维卷积函数
def twodConv(f, w, method='zero'):
    # p是经过填充后的f
    p = padding(f, w, method)
    # 待处理矩阵f的行和列
    rowf, colf = f.shape
    # 卷积核的行和列
    roww, colw = w.shape
    g = np.zeros((rowf, colf))

    # 对卷积核w进行逆时针180°翻转
    w = np.flipud(w)  # 上下翻转
    w = np.fliplr(w)  # 左右翻转

    # 滑动卷积计算
    for i in range(0, rowf):
        for j in range(0, colf):
            # value = p[i+m][j+n] * w[m][n]
            # g[i][j] += value
            g[i][j] = np.sum(np.multiply(w, p[i:(i + roww), j:(j + colw)]))

    g = g.astype('uint8')
    return g


if __name__ == '__main__':
    img = cv2.imread('/Users/zhengjiatong/Desktop/cameraman.tif', cv2.IMREAD_UNCHANGED)

    # 定义w，为3×3的矩阵
    w = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    # w = np.array([[0,0,1,0,0],[0,0,1,0,0],[1,1,-4,1,1],[0,0,1,0,0],[0,0,1,0,0]])

    g = twodConv(img, w, 'replicate')
    p = twodConv(img, w, 'zero')

    plt.figure()
    plt.subplot(1, 2, 1)
    ax = plt.gca()
    plt.axis('off')  # 去掉坐标系
    plt.imshow(p, cmap='gray')
    ax.set_title("padding zero")

    plt.subplot(1, 2, 2)
    ax = plt.gca()
    plt.axis('off')  # 去掉坐标系
    plt.imshow(g, cmap='gray')
    ax.set_title("padding method:replicate")

    plt.show()


