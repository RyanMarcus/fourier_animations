import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


# mode = "L" means grayscale
img = scipy.ndimage.imread("kar.jpg", mode="L")
fft = np.fft.rfft2(img)

# clear the DC componenet (linear shift)
fft[0][0] = 0

def gen_with_k_components(k):
    print(k)
    mag = np.abs(fft)
    phase = np.angle(fft)

    largest_k_indices = np.argpartition(mag, -k, axis=None)[-k:]

    mask = np.zeros(fft.shape)
    for flat_idx in largest_k_indices:
        mask[np.unravel_index(flat_idx, mag.shape)] = 1

    x = fft * mask
    
    ifft = np.fft.irfft2(x)
    ifft = np.real(ifft)
    return ifft


for i in range(1, 17):
    plt.subplot(4, 4, i)
    plt.title(str(2**i) + " largest components")
    plt.imshow(gen_with_k_components(20), cmap="gray")
plt.show()



