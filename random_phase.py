import numpy as np
import scipy.ndimage
import scipy.misc as smp
import matplotlib.pyplot as plt
import math
import scipy.misc
from multiprocessing import Pool
import imageio
import parallelTestModule
import glob
import os
import secrets
import shutil
import sys


try:
    from imgurpython import ImgurClient
    imgur_support = True
except ImportError:
    imgur_support = False

from uuid import uuid4
import shutil


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

def gen_image(inp, fft, user_uuid):
    idx, i = inp
    x = gen_with_k_components(i, fft)
    scipy.misc.imsave("{}/frame{}.png".format(user_uuid, idx), x)

def gen(fft, user_uuid, num=100):
    with Pool(8) as p:
        values = enumerate(np.linspace(0, 2*math.pi, num=num))
        p.starmap(gen_image, zip(values, [fft]*num, [user_uuid]*num))


def make_image_from_pixels(pixel_data, new_image_name):
    # pixel data is a JSON parsed dict 
    # Red: array of red, Blue: array of blue, Green: array of green
    # Height: height of array
    if pixel_data['height']*pixel_data['width'] != len(pixel_data['red']):
        print("Bad height and width")
        sys.exit(1)
    if pixel_data['red']!=pixel_data['blue'] or pixel_data['red'] !=pixel_data['green']:
        print("Pixel data not same size")
        sys.exit(1)
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
    

def read_image_and_convert_gif(person_uuid, image_name=None):
    
    if image_name==None:
        image_name = '{}.png'.format(person_uuid)
    gif_name = '{}.gif'.format(person_uuid)

    # Stolen from Stack Overflow
    extractor = parallelTestModule.ParallelExtractor()
    extractor.runInParallel(numProcesses=2, numThreads=4)

    # mode = "L" means grayscale
    img = scipy.ndimage.imread(image_name)
    fft = [np.fft.rfft2(img[:,:,x]) for x in range(img.shape[2])]

    # clear the DC componenet (linear shift)
    for color_channel in fft:
        color_channel[0][0] = 0
    gen(fft, person_uuid)
    print("Frames Generated: Generating GIF (Pronounced Jif)")
    with imageio.get_writer(gif_name, mode='I') as writer:
        for filename in sorted(glob.glob("{}/*.png".format(person_uuid)), key=file_key):
            image = imageio.imread(filename)
            writer.append_data(image)
            
def upload_to_imgur(image_name):
    client = ImgurClient(secrets.imgur_key, secrets.imgur_secret)
    print(image_name)
    r = client.upload_from_path(image_name)
    return r    
            
if __name__=="__main__":
    person_uuid = str(uuid4())
    os.mkdir(person_uuid)
    choice = input("Press 1 to read from pixel data, press 2 to read from an image: ")
    if choice == "1":
        values = [int(x) for x in open(input("Name of file: ")).read().split(',')]
        pixel_data = {
            'red': values,
            'blue': values,
            'green': values,
            'height': int(input("Enter height: ")),  # 520
            'width':  int(input("Enter width: ")),  # 504
        }
        make_image_from_pixels(pixel_data, person_uuid+".png")
        picture_location = None
    elif choice == "2":
        picture_location = input("Enter path for image: ")
    else:
        print("Exiting")
        sys.exit(1)

    read_image_and_convert_gif(person_uuid, picture_location)
    if imgur_support:
        should_upload = input("Upload to imgur? Y/N: ").lower() not in ["n",""]
        if should_upload:
            try:
                r = upload_to_imgur(person_uuid+'.gif')
                print("URL: {}".format(r['link']))

            except:
                print("Upload failed")
    else:
        print("Can't upload to imgur. Please do pip install imgurpython")

    shutil.move("{}.gif".format(person_uuid), "result.gif")
    print("Gif saved as result.gif")
    try:
        os.remove("{}.png".format(person_uuid))
    except FileNotFoundError:
        pass
    shutil.rmtree(person_uuid)
