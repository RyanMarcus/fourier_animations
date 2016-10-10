import numpy as np
import matplotlib.pyplot as plt
import random
import math

def pm2i(mag, phase):
    return mag * np.exp(1j * phase)

def gen_with_k_components(k):
    mag = np.zeros((500, 500))
    phase = np.zeros((500, 500))


    for i in range(k):
        for j in range(k):
            mag[i][j] = random.uniform(0, 1)
            phase[i][j] = random.uniform(-math.pi, math.pi)
        
            mag[j][i] = mag[i][j]
            phase[j][i] = phase[i][j]

    # set the DC component to zero (no offset)
    mag[0][0] = 0
    phase[0][0] = 0
    
    x = pm2i(mag, phase)
    ifft = np.fft.irfft2(x)
    ifft = np.real(ifft)
    return ifft


for i in range(1, 17):
    plt.subplot(4, 4, i)
    plt.imshow(gen_with_k_components(8), cmap="Blues")
plt.show()



