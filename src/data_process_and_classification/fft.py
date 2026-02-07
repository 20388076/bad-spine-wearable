# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 15:10:41 2025

@author: AXILLIOS
"""

import numpy as np
from scipy.fft import fft

# Signal of length 38


x = [0.192, 0.172, 0.196, 0.182, 0.196, 0.192, 0.165, 0.180,
                            0.172, 0.180, 0.194, 0.187, 0.196, 0.208, 0.182, 0.199]   # small DC + sine component
N = len(x)
n = np.arange(N)
# True FFT (exact length)
X_true = np.real(fft(x))

# Zero-padded FFT (next power of 2 = 64)
N_pad = 64
x_pad = np.zeros(N_pad)
x_pad[:N] = x
X_pad = fft(x_pad)

# Compare first 10 bins
print("k | |X_true[k]| | |X_pad[k]|")
for k in range(N):
    print(f"{k:2d} | {abs(X_true[k]):7.3f} | {abs(X_pad[k]):7.3f}")
print("true fft")
print(X_true)
#print("padded fft")
#print(X_pad)

