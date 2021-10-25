# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:49:08 2021

@author: mjeschke
"""
import numpy as np

def autocorr(spike_t, binsize, max_t, normed = True, removeCenter = False):
    """
    AUTOCORR calculates an estimate of the autocorrelation of an input spike 
    train.

    Parameters
    ----------
    spike_t : list or 1D numpy.array
        the spike train from which to calculate the autocorrelation given as
        time stamps of spikes.
    binsize : scalar 
        time duration of bins in which to evaluate the autocorrelation.
    max_t : scalar
        maximum lag (time difference) in which to calculate the autocorrelation.
    normed : BOOLEAN, optional
        whether to normalize the autocorrelation by the duration of the 
        observation window. Estimated from the maximum time stamp in the 
        spike train. The default is True.
    removeCenter: BOOLEAN, optional
        whether to remove the central bin in the autocorrelation as it is much 
        larger than the other bins.

    Returns
    -------
    ac : numpy.array
        The estimated autocorrelation of the spike train.
    xbin_edges : numpy.array
        bin edges in which the autocorrelation was estimated.

    """
    
    xbin_edges = np.arange(-max_t-binsize/2, max_t+binsize, binsize) # calculate bin edges for the autocorrelation
    ac = np.zeros(len(xbin_edges)-1) # prepare a numpy array for th autocorrelation
     
    for iSpk in range(len(spike_t)): # loop through spikes
       relative_spike_t = spike_t - spike_t[iSpk] # calculate the spike timestamps relative to the current one
     
       ac = ac + np.histogram(relative_spike_t, bins = xbin_edges)[0] # calculate a histogram of relative spike times

    if normed:
        ac = ac/max(spike_t)
        
    if removeCenter:
        center_bin = int(max_t/binsize)
        ac[center_bin:center_bin+1] = 0    
    
    return ac, xbin_edges

def crosscorr(spike_t_1, spike_t_2, binsize, max_t, normed = True):
    """
    CROSSCORR calculates an estimate of the crosscorrelation of an input spike 
    train.

    Parameters
    ----------
    spike_t_1 : list or 1D numpy.array
        the first spike train from which to calculate the crosscorrelation 
        given as time stamps of spikes.
    spike_t_2 : list or 1D numpy.array
        the first spike train from which to calculate the crosscorrelation 
        given as time stamps of spikes.    
    binsize : scalar 
        time duration of bins in which to evaluate the crosscorrelation.
    max_t : scalar
        maximum lag (time difference) in which to calculate the crosscorrelation.
    normed : BOOLEAN, optional
        whether to normalize the crosscorrelation. The default is True.

    Returns
    -------
    ac : numpy.array
        The estimated crosscorrelation of the spike train.
    xbin_edges : numpy.array
        bin edges in which the crosscorrelation was estimated.

    """
    
    xbin_edges = np.arange(-max_t-binsize/2, max_t+binsize, binsize) # calculate bin edges for the autocorrelation
    cc = np.zeros(len(xbin_edges)-1) # prepare a numpy array for th autocorrelation
     
    for cur_spk in spike_t_1:  # loop through spikes
     
       relative_spike_t = spike_t_2 - cur_spk # calculate the spike timestamps relative to the current one
       
       cc = cc + np.histogram(relative_spike_t, bins = xbin_edges)[0] # calculate a histogram of relative spike times
       
    if normed:
        cc = cc/max(cc) # normalize
    
    return cc, xbin_edges

if __name__ == '__main__':
    # in standalone mode we want to check whether autocorrelation/crosscorrelation
    # code is correct visually based on mock example of a periodic spike train
    # AUTOCORRELATION: we expect peaks only at multiple integers of the period
    # and the throughs have to be zero
    # the drop off between the central peak and the next peaks should be 1 over
    # the total number of spikes
    import matplotlib.pyplot as plt # import plotting library

    # general parameters
    dt = 0.001   # time step in seconds
    t  = [0, 10] # time interval (length) of spike train to generate  
    bin_size = 0.001 # in s
    max_t    = 1 # maximum lag for correlation in s
    
    # generate a mock spike train with periodic time stamps
    period = 50/1000 # in s
    spk_t = np.arange(t[0], t[1], period) # the spike train
    
    # periodic spike train auto correlation function
    ac_mock, bin_edges = autocorr(spk_t, bin_size, max_t) # calculate the autocorrelation function
    
    # generate a second mock spike train with periodic time stamps as the first
    # spike train but shifted by half a period
    shift = period/5
    spk_t_2 = np.arange(t[0], t[1], period) + period/5 # the spike train

    # periodic spike train auto correlation function
    cc_mock, bin_edges = crosscorr(spk_t, spk_t_2, bin_size, max_t) # calculate the autocorrelation function

    # Plot the data
    h_fig, h_ax = plt.subplots(ncols = 2, nrows = 1)
    h_ax[0].vlines([spk_t, spk_t],-1, -0.5, Color = [0, 0, 0],  label=f"Period: {period} s") # note, plots all spikes in one command
    h_ax[0].set_ylim([-1.5, 0])
    h_ax[0].set_xlim([0, 10])
    h_ax[0].set_yticks([])
    h_ax[0].set_ylabel('spikes')
    h_ax[0].set_xlabel('time (s)')
    
    h_ax[0].legend()
    
    h_ax[1].plot(bin_edges[:-1]+bin_size/2, ac_mock)
    
    h_ax[1].set_xlabel('lag [s]')
    h_ax[1].set_ylabel('acorr')
    
    h_fig.suptitle('Autocorrelation of a periodic spike train')
    
    # Plot the data
    h_fig, h_ax = plt.subplots(ncols = 2, nrows = 1)
    h_ax[0].vlines([spk_t, spk_t],-1, -0.5, Color = [0.1, 0.1, 0.5], label=f"Period: {period} s") # note, plots all spikes in one command
    h_ax[0].vlines([spk_t_2, spk_t_2],0, 0.5, Color = [0, 0, 0], label=f"shifted by {shift} s")

    h_ax[0].set_ylim([-1.5, 1.0])
    h_ax[0].set_xlim([0, 10])
    h_ax[0].set_yticks([])
    h_ax[0].set_ylabel('spikes')
    h_ax[0].set_xlabel('time (s)')
    
    h_ax[0].legend()
    
    h_ax[1].plot(bin_edges[:-1]+bin_size/2, cc_mock)
    
    h_ax[1].set_xlabel('lag [s]')
    h_ax[1].set_ylabel('crosscorr')
    
    h_fig.suptitle('Crosscorrelation of two periodic spike trains with same period but phase shifted')

    plt.show()