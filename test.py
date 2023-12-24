import cv2
import numpy as np
def normalization(data):
    dst = np.zeros(data.shape, dtype=np.float64)
    a = cv2.normalize(data,dst,0, 1, cv2.NORM_MINMAX)
    return a
def dft2D(f):
    h,w = f.shape
    F = np.zeros(f.shape, dtype=complex)
    for i in range(h):
        F[i,:] = np.fft.fft(f[i,:])
    for i in range(w):
        F[:,i] = np.fft.fft(F[:,i])
    return F

def idft2D(F):
    h,w = F.shape
    F1 = np.conj(F)
    f = np.zeros(F1.shape,dtype=complex)
    for i in range(h):
        f[i,:] = np.fft.fft(F1[i,:])
    for i in range(w):
        f[:,i] = np.fft.fft(f[:,i])
    f = f/(h*w)
    f = np.conj(f)
    f = np.abs(f)
    return f


f1= cv2.imread("/Users/zhengjiatong/Desktop/rose512.tif",cv2.IMREAD_GRAYSCALE)
# dst = np.zeros(f1.shape, dtype=np.float64)
# f=cv2.normalize(f1, dst, 0, 1, cv2.NORM_MINMAX)
# f = normalization(f1)
# F = dft2D(f)
# g = idft2D(F)
# d = np.abs(f-g)
# cv2.imshow('original', f1)
# # cv2.imshow('f',f)
# cv2.imshow('g', g)
# cv2.imshow('d', d)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
