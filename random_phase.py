import numpy as np
import scipy.ndimage
import scipy.misc as smp
import matplotlib.pyplot as plt
import random
import math
import scipy.misc
from multiprocessing import Pool
import imageio
import parallelTestModule
import glob
import os
import flask



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
		
def make_image_from_pixels(pixel_data, new_image_name):
    # pixel data is a JSON parsed dict 
    # Red: array of red, Blue: array of blue, Green: array of green
    # Height: height of array
    new_image = np.zeros( (pixel_data['height'],pixel_data['width'],3), dtype=np.uint8 )
    combined_rgb = list(zip(pixel_data['red'], pixel_data['blue'], pixel_data['green']))
    #print (combined_rgb)
    i = 0
    for row in new_image:
        for column in row:
            for color in range(3):
                column[color]=combined_rgb[i][color]
            i+=1
    img = smp.toimage( new_image )       # Create a PIL image
    img.save(new_image_name, "PNG")
    #img.show()
    

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
    values = [int(x) for x in open('data.txt').read().split(',')]
    pixel_data = {
        'red':values, 
        'blue':values,
        'green': values,
        'height': 520,
        'width': 504,
    }
    make_image_from_pixels(pixel_data, 'crap.png')
    read_image_and_convert_gif('crap.png')