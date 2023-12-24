# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
def dft2D(f):
    h,w = f.shape
    F = np.zeros(f.shape, dtype=complex)
    for i in range(h):
        F[i,:] = np.fft.fft(f[i,:])
    for i in range(w):
        F[:,i] = np.fft.fft(F[:,i])
    return F
dft2D()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
