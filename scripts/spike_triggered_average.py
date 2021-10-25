# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:22:02 2021

@author: mjeschke
"""
import numpy as np


def spike_triggered_average(spike_t, stimulus, dt, Tmin, Tmax):
    """
    SPIKE_TRIGGERED_AVERAGE calculates the spike triggered average (STA) of a 
    signal within a chosen window. It assumes that stimulus starts at time 0.

    Parameters
    ----------
    spike_t : list or 1D numpy.array
        the spike train from which to calculate the STA, given as
        time stamps of spikes.
    stimulus : list or 1D numpy.array
        a temporally variable parameter that describes the time course of the 
        stimulus.
    dt : scalar
        is the sampling interval of the stimulus.
    Tmin : scalar
        describe a window before a spike in which to look at the 
        stimulus, are scalar variables.
    Tmax : scalar
        describe a window before a spike in which to look at the 
        stimulus.
        
        all parameters are to be given in seconds or at least the same unit

    Returns
    -------
    sta : numpy.array
        the spike triggered average.
    x : numpy.array
        time vector for the STA.
    non_averaged_spikes : TYPE
        spike timestamps not used for the STA b/c not enough stimulus data was
        available.

    """  

    x = np.arange(Tmin, Tmax, dt)[::-1]*-1 # time vector for the stimulus average
    nBins_sta = x.shape[0]
    
    sta = np.zeros(nBins_sta) # prepare an array to contain the spike triggered average
    
    # loop through spikes, creating the average by first summing stimulus snippets and then
    # dividing by number of spikes used
    non_averaged_spikes = []
    for curSpike in spike_t:
         # is enough data of the stimulus available for the given spike?
            if (curSpike - Tmax >= 0 and curSpike - Tmin <= len(stimulus)*dt):
                start_bin = int(np.floor((curSpike - Tmax)/dt))

                sta = np.add(sta, stimulus[start_bin:start_bin + nBins_sta])
            else: # keep track of spikes that did not enter the averaging procedure
                non_averaged_spikes.append(curSpike)
                                
    # normalize by the number of spikes used for averaging                                
    sta = sta / (len(spike_t) - len(non_averaged_spikes))
                                
    return sta, x, non_averaged_spikes

if __name__ == '__main__':
    # in standalone mode we want to check whether STA code is correct visually
    import matplotlib.pyplot as plt # import plotting library
    
    # test STA with a mock example
    # generate sinusoidal signal plus noise
    f = 50 # Hz
    dt = 1/1000 # 1000 Hz sampling
    dur = 4
    t = np.arange(0,dur,dt)
    signal = np.sin(2*np.pi*f*t) + np.random.rand(t.shape[0])*5 # sinusoid plus a much larger noise
    signal *= 1/max(abs(signal)) # normalize for visualization
    # generate a spike train always firing at period of sinusoid
    spike_t = np.arange(0,dur,1/f) + 5/(f * 4)
    
    # now check the spike triggered average
    Tmin, Tmax = 0.0, 1/f # let us use one full period of the stimulus
    sta, t_sta, non_averaged_spikes = spike_triggered_average(spike_t, signal,\
                                                              dt, Tmin, Tmax)
    
    # visualize
    h_fig, h_ax = plt.subplots(nrows = 1, ncols = 2)
    h_ax[0].plot(np.arange(0, dur, dt), signal)
    h_ax[0].eventplot(spike_t, lineoffsets = [1.5], color = 'gray')
    
    h_ax[1].plot(t_sta * 1000, sta)
    
    plt.setp(h_ax[0], xlim = [0, dur], yticks = [0.5, 1.5], \
             yticklabels = ['signal', 'spikes'], xlabel = 'time (s)')
    plt.setp(h_ax[1], xlabel = 'time before a spike (ms)', \
             ylabel = 'signal amplitude')
    
    plt.show()
    
    
    
    