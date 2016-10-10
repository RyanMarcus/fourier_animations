import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

import random
import math

# mode = "L" means grayscale
img = scipy.ndimage.imread("image.jpg", mode="L")
fft = np.fft.rfft2(img)

# clear the DC componenet (linear shift)
fft[0][0] = 0

def gen_with_k_components(k, num_phases):
    k = int(k / num_phases)
    k = max(k, 1)
    phase = np.angle(fft)

    # pick a phase
    mask = np.zeros(fft.shape)
    for selected_phase in range(num_phases):
        mag = np.abs(fft)
        phase_increm = (2*math.pi) / num_phases
        phase_floor = (phase_increm * selected_phase) - math.pi
        phase_ceil = (phase_increm * (selected_phase + 1)) - math.pi
        print(phase_floor, phase_ceil)
        ineligable = np.where((phase >= phase_ceil) | (phase <= phase_floor))
        mag[ineligable] = 0
        
        largest_k_indices = np.argpartition(mag, -k, axis=None)[-k:]

        for flat_idx in largest_k_indices:
            mask[np.unravel_index(flat_idx, mag.shape)] = 1

    x = fft * mask

    ifft = np.fft.irfft2(x)
    ifft = np.real(ifft)
    return ifft


for i in range(1, 17):
    plt.subplot(4, 4, i)
    plt.title(str(2**i) + "/8 largest components from 1/8th of phase circle")
    plt.imshow(gen_with_k_components(2**i, 8), cmap="gray")
plt.show()



