import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import random
import math
import scipy.misc
from multiprocessing import Pool

def pm2i(mag, phase):
    return mag * np.exp(1j * phase)

# mode = "L" means grayscale
img = scipy.ndimage.imread("image.jpg", mode="L")
fft = np.fft.rfft2(img)

# clear the DC componenet (linear shift)
fft[0][0] = 0

def gen_with_k_components(k):
    mag = np.abs(fft)
    phase = np.angle(fft)

    largest_k_indices = np.argpartition(mag, -10, axis=None)[-10:]

    mask = np.zeros(fft.shape)
    for flat_idx in largest_k_indices:
        mask[np.unravel_index(flat_idx, mag.shape)] = k

    phase += mask
    x = pm2i(mag, phase)
    
    ifft = np.fft.irfft2(x)
    ifft = np.real(ifft)
    return ifft

def gen_image(inp):
    idx, i = inp
    scipy.misc.imsave("frames/frame{}.png".format(idx), gen_with_k_components(i))

def gen():
    with Pool(4) as p:
        values = enumerate(np.linspace(0, 2*math.pi, num=100))
        p.map(gen_image, values)

gen()

