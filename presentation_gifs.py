# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 18:17:11 2021


create gifs for the W2 Oldenburg talk

@author: mjeschke
"""

import os
import numpy as np

from copy import copy

import itertools
from progress.bar import Bar # a nice looking progress bar for running in command line mode

import matplotlib.pyplot as plt
import imageio

eps = np.finfo(np.float32).eps # machine precision


# Build GIF
def createGIF(gifName, imageList, nFrame, folder = None, barFlag = True):
    
    
    
    with imageio.get_writer(gifName, mode='I') as writer:
        if barFlag:
            bar = Bar('GIF Frame', max = nFrame)
            
        for filename in imageList:
            image = imageio.imread(filename)
            writer.append_data(image)
            
            if barFlag:
                bar.next()
                
        if barFlag:
            bar.finish()
            


# create convolution gif
tmin, tmax, tstep = -2, 2, 0.1
t = np.arange(tmin, tmax+tstep/2, tstep)

def box(t, tOn_Off = 1, amplitude = 1):
    
    if not isinstance(t, (list, np.ndarray)):
        t = np.array([t])
    
    a = np.zeros(len(t))
    a[(t>= -tOn_Off-eps ) & (t<=tOn_Off+eps)] = amplitude
    
    return a

a = lambda t: box(t, .5, 1)
b = lambda t: box(t, .45, 1)

# make a real rectangular box for plotting
def makeRectBox(t, box):
    
    timeBox = copy(t)
    timeBox = np.insert(timeBox, [np.where(box == 1)[0][0], np.where(box == 1)[0][-1]], [t[np.where(box == 1)[0][0] ], t[np.where(box == 1)[0][-1]]])
    box = np.insert(box, [np.where(box == 1)[0][0], np.where(box == 1)[0][-1]+1], [0, 0])
    
    return timeBox, box

def convolution(x):
  fog=0
  xt=-3.05
  dx=0.1
  while xt<3.05: 
    fog=fog+a(xt)*b(x-xt)*dx
    xt=xt+dx
  return(fog[0])


box_convolved = [convolution(cur_time) for cur_time in t]

box_convnp = np.convolve(a(t), b(t), 'same')

h_fig, h_ax = plt.subplots(nrows = 2, ncols = 1)
h_ax[0].plot(t, a(t), '.-', label = '$f(t)$')

h_ax[0].plot(t-t[-1], b(t), '.-', label = '$h(t)$')

h_ax[0].grid()

h_ax[1].plot(t, box_convolved, '.-', label = '$g(t) = f(t)*h(t)$', color = 'gray')
h_ax[1].grid()

# collect labels for a common legend
lines_labels = [ax.get_legend_handles_labels() for ax in h_fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

h_ax[1].legend(lines, labels)

plt.setp(h_ax, xlim = [tmin, tmax])
# plt.plot(t, box_convnp, '.-', color = 'gray')


# Now animate the convolution
frameName = (f".\dummy\convolFrame_{idx}.png" for idx in range(len(t)))
frameName_create, frameName_save, frameName_delete = itertools.tee(frameName, 3)
animationName = 'convolution.gif'

# boxA = a(t) # make a real rectangular box
# timeA = copy(t)
# timeA = np.insert(timeA, [np.where(boxA == 1)[0][0], np.where(boxA == 1)[0][-1]], [t[np.where(boxA == 1)[0][0] ], t[np.where(boxA == 1)[0][-1]]])
# boxA = np.insert(boxA, [np.where(boxA == 1)[0][0], np.where(boxA == 1)[0][-1]+1], [0, 0])

timeA, boxA = makeRectBox(t, a(t))

h_fig, h_ax = plt.subplots(nrows = 2, ncols = 1)
h_ax[0].plot(timeA, boxA, '-', label = 'f(t)')

h_ax[1].grid()
h_ax[0].grid()

h_ax[0].set_ylabel('Amplitude')
h_ax[1].set_xlabel('Time')
h_ax[1].set_ylabel('Amplitude')

plt.setp(h_ax, xlim = [tmin, tmax], ylim = [0, 1.1])

bar = Bar('Frame', max = len(t))
for curIdx, curTime in enumerate(t):
    # plot the shifting second box
    
    timeB, boxB = makeRectBox(t, b(t-curTime))
    h_curve1, = h_ax[0].plot(timeB, boxB, '-', color = 'orange', label = '$h(t)$')
      
    # plot the resulting convolved function
    h_curve2, = h_ax[1].plot(t[:curIdx:], box_convolved[:curIdx:], '-', color = 'gray', label = '$g(t) = f(t) \ast h(t)$')
    
    # collect labels for a common legend
    lines_labels = [ax.get_legend_handles_labels() for ax in h_fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    
    h_ax[1].legend(lines, labels)
        
    # save the current frame as a png
    plt.savefig(next(frameName_create), dpi = 300, transparent=True) 
    
    # remove the updated curves
    h_curve1.remove()
    h_curve2.remove()

    # progress bar
    bar.next()
    
    # print(f"Frame: {curIdx}/{len(t)}")
    
bar.finish()
    
# save the frames as a gif
print('Creating GIF')
createGIF(animationName, frameName_save, len(t))    

# delete unnecessary pngs
print('Deleting Frames')
for curFrame in frameName_delete:
    os.remove(curFrame)
