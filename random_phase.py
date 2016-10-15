import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import random
import math
import scipy.misc
from multiprocessing import Pool
import imageio
import parallelTestModule
import glob
import os


# mode = "L" means grayscale
img = scipy.ndimage.imread("flower.jpg")
fft = [np.fft.rfft2(img[:,:,x]) for x in range(img.shape[2])]

def file_key(path):
		fname = os.path.split(path)[1]
		number = fname.strip("frame").strip(".png")
		return int(number)
	
def pm2i(mag, phase):
    return mag * np.exp(1j * phase)


def gen_with_k_components(k):
    toR = []
    for color_channel_fft in fft:
        mag = np.abs(color_channel_fft)
        phase = np.angle(color_channel_fft)

        num_components = 500
        largest_k_indices = np.argpartition(mag,
                                            -num_components,
                                            axis=None)[-num_components:]

        mask = np.zeros(color_channel_fft.shape)
        for flat_idx in largest_k_indices:
            mask[np.unravel_index(flat_idx, mag.shape)] = k

        phase += -k #mask
        x = pm2i(mag, phase)
        #same as: x = color_channel_fft * np.exp(1j * k)
    
        ifft = np.fft.irfft2(x)
        ifft = np.real(ifft)
        toR.append(ifft)
    return np.dstack(toR)

def gen_image(inp):
    idx, i = inp
    x = gen_with_k_components(i)
    scipy.misc.imsave("frames/frame{}.png".format(idx), x)

def gen():
    with Pool(8) as p:
        values = enumerate(np.linspace(0, 2*math.pi, num=100))
        p.map(gen_image, values)

if __name__=="__main__":
    extractor = parallelTestModule.ParallelExtractor()
    extractor.runInParallel(numProcesses=2, numThreads=4)


    # clear the DC componenet (linear shift)
    for color_channel in fft:
        color_channel[0][0] = 0
    gen()
    with imageio.get_writer('movie.gif', mode='I') as writer:
        for filename in sorted(glob.glob("frames/*.png"), key=file_key):
            image = imageio.imread(filename)
            writer.append_data(image)