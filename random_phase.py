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



def file_key(path):
		fname = os.path.split(path)[1]
		number = fname.strip("frame").strip(".png")
		return int(number)
	
def pm2i(mag, phase):
    return mag * np.exp(1j * phase)


def gen_with_k_components(k, fft):
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

def gen_image(inp, fft):
    idx, i = inp
    x = gen_with_k_components(i, fft)
    scipy.misc.imsave("frames/frame{}.png".format(idx), x)

def gen(fft, num=100):
    with Pool(8) as p:
        values = enumerate(np.linspace(0, 2*math.pi, num=num))
        p.starmap(gen_image, zip(values, [fft]*num))

def read_image_and_convert_gif(image_name):
    extractor = parallelTestModule.ParallelExtractor()
    extractor.runInParallel(numProcesses=2, numThreads=4)

	# mode = "L" means grayscale
    img = scipy.ndimage.imread(image_name)
    fft = [np.fft.rfft2(img[:,:,x]) for x in range(img.shape[2])]

    # clear the DC componenet (linear shift)
    for color_channel in fft:
        color_channel[0][0] = 0
    gen(fft)
    print("Frames Generated: Generating GIF (Pronounced Jif)")
    with imageio.get_writer('movie.gif', mode='I') as writer:
        for filename in sorted(glob.glob("frames/*.png"), key=file_key):
            image = imageio.imread(filename)
            writer.append_data(image)
			
			
if __name__=="__main__":
    read_image_and_convert_gif('flower.jpg')