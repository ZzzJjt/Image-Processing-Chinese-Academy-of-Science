import cv2
import numpy as np
import matplotlib.pyplot as plt
# 调用second.py文件中的rgb1gray函数
# from ColorBlackWhite import rgb1gray
# 调用third.py文件中的twodConv函数
from ImageConvolutionalFunction  import twodConv
# 调用fourth.py文件中的gaussKernel函数
from NormalizedTwoDimensionGaussianFilteringKernelFunction import gaussKernel


def rgb1gray(f, method='NTSC'):
    # 防止三通道相加溢出，所以先转换类型
    f = f.astype(np.float32) / 255

    # 获取图像的三通道
    # 法一用cv：b,g,r = cv2.split(f)
    # 法二自己分离：
    b = f[:, :, 0]
    g = f[:, :, 1]
    r = f[:, :, 2]
    if method == 'average':
        gray = (r + g + b) / 3
    elif method == 'NTSC':
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # 再把类型转换回来
    gray = (gray * 255).astype('uint8')

    return gray
if __name__ == '__main__':
    img1 = cv2.imread('/Users/zhengjiatong/Desktop/cameraman.tif',cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread('/Users/zhengjiatong/Desktop/einstein.tif',cv2.IMREAD_UNCHANGED)
    #待转为NTSC灰度图像
    img3 = cv2.imread('/Users/zhengjiatong/Desktop/mandril_color.tif',cv2.IMREAD_UNCHANGED)
    img4 = cv2.imread('/Users/zhengjiatong/Desktop/lena512color.tiff',cv2.IMREAD_UNCHANGED)
    #转换后的灰度图像
    img3Final = rgb1gray(img3, 'NTSC')
    img4Final = rgb1gray(img4, 'NTSC')

    # method = 'replicate'
    # #method = 'zero'
    #
    # #sig = [1,2,3,5]
    #
    # w = gaussKernel(5)
    # # 在这个函数内部要转为uint8，否则在0-1的范围内显示；若超过1就是白色了
    # g = twodConv(img1, w, method)
    # cv2.imshow('sigma=5', g)
    # cv2.waitKey(0)

    # # 1.对四幅图像 分别采用σ=1，2，3，5的高斯滤波
    # # plt.figure()
    # name_list = ['σ=1', 'σ=2', 'σ=3', 'σ=5']
    # for i in range(0, 4):
    #     w = gaussKernel(sig[i])
    #     p = twodConv(img1, w, method)
    #     plt.subplot(2,2,i+1)
    #     # ax = plt.gca()
    #     plt.axis('off') #去掉坐标系
    #     plt.imshow(p, cmap='gray')
    #     #ax.set_title("σ=", sig[i])
    # plt.show()

    #2.在σ=1时把自己写的函数与直接调用函数的结果作比较
    sig = 1
    m = 6 * sig + 1
    w = gaussKernel(1, m)
    res1 = twodConv(img4Final, w, 'replicate')

    #(m, m)表示高斯矩阵的长与宽，标准差取1
    res = cv2.GaussianBlur(img4Final, (m, m), 1)

    cv2.imshow('lena512.tiff',np.abs(res1.astype(int)-res.astype(int)).astype('uint8'))
    cv2.waitKey(0)

    # # 3.比较两幅图像在像素复制和补零下滤波结果在边界上的差别
    # method1 = 'replicate'
    # method2 = 'zero'
    # # 保证其他条件不变
    # sig = 1
    # m = 6 * sig + 1
    # w = gaussKernel(1, m)
    # firstRe = twodConv(img1, w, method1)
    # firstZe = twodConv(img1, w, method2)
    # secondRe = twodConv(img2, w, method1)
    # secondZe = twodConv(img2, w, method2)
    #
    # plt.figure()
    # plt.subplot(221)
    # ax = plt.gca()
    # plt.axis('off')  # 去掉坐标系
    # plt.imshow(firstRe, cmap='gray')
    # ax.set_title("cameraman.tif replicate")
    #
    # plt.subplot(222)
    # ax = plt.gca()
    # plt.axis('off')  # 去掉坐标系
    # plt.imshow(firstZe, cmap='gray')
    # ax.set_title("cameraman.tif zero")
    #
    # plt.subplot(223)
    # ax = plt.gca()
    # plt.axis('off')  # 去掉坐标系
    # plt.imshow(secondRe, cmap='gray')
    # ax.set_title("einstein.tif replicate")
    #
    # plt.subplot(224)
    # ax = plt.gca()
    # plt.axis('off')  # 去掉坐标系
    # plt.imshow(secondZe, cmap='gray')
    # ax.set_title("einstein.tif zero")
    #
    # plt.show()




