# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.

import numpy as np
import cv2
import matplotlib.pyplot as plt

def scanLine4e(f, I, loc):
    if loc == 'row':
        return f[I]
    elif loc == 'column':
        return f[:,I]

if __name__ == '__main__':
    #首先读入两幅图像
    img1 = cv2.imread('/Users/zhengjiatong/Desktop/cameraman.tif',cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread('/Users/zhengjiatong/Desktop/einstein.tif',cv2.IMREAD_UNCHANGED)

    #得到图像的行数和列数
    sp1 = img1.shape
    row1 = sp1[0]
    column1 = sp1[1]
    #当行数或列数为整数时，中心点不唯一，任取其一
    rowCenter1 = row1//2
    columnCenter1 = column1//2

    sp2 = img2.shape
    row2 = sp2[0]
    column2 = sp2[1]
    rowCenter2 = row2//2
    columnCenter2 = column2//2

    #调用scanLine4e函数得到中心行和中心列的像素灰度序列
    rowData1 = scanLine4e(img1, rowCenter1, 'row')
    columnData1 = scanLine4e(img1, columnCenter1, 'column')

    rowData2 = scanLine4e(img2, rowCenter2, 'row')
    columnData2 = scanLine4e(img2, columnCenter2, 'column')

    #画图-subplot把四个图放在一张图上
    plt.figure(figsize=(15,15))
    plt.subplot(2,2,1)
    ax = plt.gca()
    plt.plot(rowData1,color = 'green')
    ax.set_xlabel('Pixel Number')
    ax.set_ylabel('Gray Value')
    ax.set_title("cameraman.tif's row")

    plt.subplot(2,2,2)
    ax = plt.gca()
    plt.plot(columnData1,color = 'red')
    ax.set_xlabel('Pixel Number')
    ax.set_ylabel('Gray Value')
    ax.set_title("cameraman.tif's column")

    plt.subplots_adjust(hspace = 0.5)

    plt.subplot(2,2,3)
    ax = plt.gca()
    plt.plot(rowData2,color = 'green')
    ax.set_xlabel('Pixel Number')
    ax.set_ylabel('Gray Value')
    ax.set_title("einstein.tif's row")

    plt.subplot(2,2,4)
    ax = plt.gca()
    plt.plot(columnData2,color = 'red')
    ax.set_xlabel('Pixel Number')
    ax.set_ylabel('Gray Value')
    ax.set_title("einstein.tif's column")

    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
