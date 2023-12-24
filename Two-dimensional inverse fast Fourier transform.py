import numpy
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
