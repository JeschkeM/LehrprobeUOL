# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:48:15 2021

@author: mjeschke
"""
# NECESSARY PYTHON IMPORTS
import numpy as np
from matplotlib import pyplot as plt

from IPython.display import display 
import ipywidgets as widgets 
from ipywidgets import interact, Layout

from copy import copy

epsilon = np.finfo(float).eps # to take care of binning issues due to machine precision

# default size of figures generated in the Ipython notebook
plt.rcParams['figure.figsize'] = [14, 5]
# load the data
stimulus_frequencies = np.loadtxt('..//data' + '/' + 'M13W1671_stimuli.txt', skiprows = 1) 

trials_spike_times   = np.loadtxt('..//data' + '/' + 'M13W1671_trials_spike_times.txt', delimiter = ',',skiprows = 1) 

# general parameters
stimulus_duration = 0.25

trains_to_plot = []
for curTrial in np.unique(trials_spike_times[:,2]):
    trains_to_plot.append(trials_spike_times[trials_spike_times[:,2] == curTrial,3])
    
plt.figure(figsize=(6,4))
plt.plot(trials_spike_times[:,3], trials_spike_times[:,2], linestyle = 'None', marker = '.', color='gray')
plt.xlim([0, stimulus_duration])
plt.ylabel('frequency [kHz]')
plt.xlabel('time (s)')

# put a useful label on the ordinate
yticks = plt.gca().get_yticks().astype(int)
yticks = yticks[(yticks >= 0) & (yticks < len(stimulus_frequencies))]

plt.setp(plt.gca(), yticks = yticks, yticklabels = (np.round(stimulus_frequencies[yticks]/1000,2)))

plt.tight_layout()

 # save the current frame as a png
plt.savefig('M13W1671_Raster.png', dpi = 150, transparent=True) 

# Plotting
h_fig, h_ax = plt.subplots(nrows = 2, ncols = 2, figsize=(6,6), gridspec_kw = {'height_ratios':[10,1]})
h_ax[0][0].eventplot(trains_to_plot, colors='k', linelengths=0.9, linewidths=1)
h_ax[0][0].set_xlim([0, stimulus_duration])
h_ax[0][0].set_ylabel('trial #')
h_ax[0][0].set_xlabel('time a(s)')

h_ax[0][1].plot(trials_spike_times[:,3], trials_spike_times[:,2], linestyle = 'None', marker = '.', color='gray')
h_ax[0][1].set_ylabel('frequency [kHz]')
h_ax[0][1].set_xlabel('time (s)')

# put a useful label on the ordinate
yticks = h_ax[0][1].get_yticks().astype(int)
yticks = yticks[(yticks >= 0) & (yticks < len(stimulus_frequencies))]

plt.setp(h_ax[0][1], yticks = yticks, yticklabels = (np.round(stimulus_frequencies[yticks]/1000,2)))
plt.setp(h_ax[0], xlim = [0, stimulus_duration]) # focus on stimulus duration

[cur_ax.axis('off') for cur_ax in h_ax[1]] # hack as RISE.js renders weirdly

plt.tight_layout()

plt.show()