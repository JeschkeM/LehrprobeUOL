# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 16:27:36 2021

@author: mjeschke
"""

import numpy as np

sigma = 0.005
bandwidth = 0.020
mu = 0

Tmin, Tmax = -0.200, 0.200
binsize = 0.001

def gaussian_kernel(sigma, bandwidth, dt, Tmin, Tmax):
    
    x = np.arange(Tmin, Tmax+dt, dt) 
    
    x1 = sigma * np.sqrt(2 * np.pi)
    x2 = np.exp(-(x - mu)**2/(2 * sigma**2))
    gaussian_filter = (1/x1)*x2

    bandwidth_x = np.all((x > -bandwidth/2, x < bandwidth/2),0)
    gaussian_kernel = gaussian_filter[bandwidth_x] 
    
    return gaussian_kernel, x[bandwidth_x]

x = np.arange(Tmin, Tmax+binsize, binsize)

x1 = sigma * np.sqrt(2 * np.pi)
x2 = np.exp(-(x - mu)**2/(2 * sigma**2))
gaussian_filter = (1/x1)*x2

bandwidth_x = np.all((x > -bandwidth/2, x < bandwidth/2),0)
gaussian_filter_truncated = gaussian_filter[bandwidth_x]


gauss_kern, gauss_x = gaussian_kernel(sigma, bandwidth, binsize, Tmin, Tmax)


from matplotlib import pyplot as plt

plt.plot(x, gaussian_filter)
plt.plot(x[bandwidth_x], gaussian_filter_truncated)
plt.plot(gauss_x, gauss_kern, ':')


plt.gca().set_xlim(Tmin, Tmax)