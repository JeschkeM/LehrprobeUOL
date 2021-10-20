# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:40:05 2021

@author: mjeschke


MATLAB - Autocorrelation function  https://rcweb.dartmouth.edu/~mvdm/wiki/doku.php?id=analysis:course-w16:week9
function [ac,xbin] = acf(spk_t,binsize,max_t)
% function [ac,xbin] = acf(spike_times,binsize,max_t)
%
% estimate autocorrelation of input spike train
%
% INPUTS:
% spike_times: [1 x nSpikes] double of spike times (in s)
% binsize: acf bin size in s
% max_t: length of acf in s
%
% OUTPUTS:
% ac: autocorrelation estimate (normalized so that acf for the zero bin is
% 0)
% xbin: bin centers (in s)
%
% MvdM 2013
 
xbin_centers = -max_t-binsize:binsize:max_t+binsize; % first and last bins are to be deleted later
ac = zeros(size(xbin_centers));
 
for iSpk = 1:length(spk_t)
 
   relative_spk_t = spk_t - spk_t(iSpk);
 
   ac = ac + hist(relative_spk_t,xbin_centers); % note that hist() puts all spikes outside the bin centers in the first and last bins! delete later.
 
end
 
xbin = xbin_centers(2:end-1); % remove unwanted bins
ac = ac(2:end-1);
 
ac = ac./max(ac); % normalize


RECODED TO MAP NUMPY HIST IMPLEMENTATION USING BIN EDGES INSTEAD OF CENTERS 
function [ac,xbin] = acf(spk_t,binsize,max_t)
% function [ac,xbin] = acf(spike_times,binsize,max_t)
%
% estimate autocorrelation of input spike train
%
% INPUTS:
% spike_times: [1 x nSpikes] double of spike times (in s)
% binsize: acf bin size in s
% max_t: length of acf in s
%
% OUTPUTS:
% ac: autocorrelation estimate (normalized so that acf for the zero bin is
% 0)
% xbin: bin edges (in s)
%
% MvdM 2013
%
%
% MJ 2021 - modified to use bin edges with histcounts function to map onto
%           numpy implementation
 
xbin_edges = -max_t-binsize/2:binsize:max_t+binsize;
ac = zeros(1,size(xbin_edges,2)-1);
 
for iSpk = 1:length(spk_t)
 
   relative_spk_t = spk_t - spk_t(iSpk);
 
%    ac = ac + hist(relative_spk_t,xbin_centers); % note that hist() puts all spikes outside the bin centers in the first and last bins! delete later.
   
   ac = ac + histcounts(relative_spk_t, xbin_edges); 
end
 
xbin = xbin_edges;
 
ac = ac./max(ac); % normalize
"""

import numpy as np

def acf(spike_t, binsize, max_t, normed = True):
    
    if not isinstance(spike_t, np.ndarray):
        spike_t = np.array(spike_t)
    
    xbin_edges = np.arange(-max_t-binsize/2, max_t+binsize, binsize) # last bin includes the edge
    ac = np.zeros(len(xbin_edges)-1)
     
    for iSpk in range(len(spike_t)):
     
       relative_spike_t = spike_t - spike_t[iSpk]
     
       ac = ac + np.histogram(relative_spike_t, bins = xbin_edges)[0] # note that hist() puts all spikes outside the bin centers in the first and last bins! delete later.
     
    xbin = xbin_edges # remove unwanted bins
    ac = ac

    if normed:
        ac = ac/max(ac) # normalize
    
    return ac, xbin
    
    
if __name__ == '__main__':
    
    dt = 0.001
    t = [0, 10] # time interval (length) of spike train to generate
    tvec = np.arange(t[0], t[1], dt)
     
    pspike = 0.5 # probability of generating a spike in bin
    np.random.seed(10)# reset random number generator to reproducible state
    spk_poiss = np.random.rand(len(tvec)) # random numbers between 0 and 1
    spk_poiss_idx = spk_poiss < pspike # boolean array of bins with spike
    spk_poiss_t = tvec[spk_poiss_idx] # use idxs to get corresponding spike time
    
    import matplotlib.pyplot as plt
     
    plt.vlines([spk_poiss_t, spk_poiss_t],-1, -0.5, Color = [0, 0, 0]) # note, plots all spikes in one command
    ax = plt.gca()
    ax.set_ylim([-1.5, 5])
    ax.set_xlim([0, 0.1])
    ax.set_yticks([])
    
    
    # poisson spike train auto correlation function
    rate = 0.47 # in Hz
    pspike = rate*dt
    
    pspike = 0.5 # probability of generating a spike in bin
    np.random.seed(10)# reset random number generator to reproducible state
    spk_poiss = np.random.rand(len(tvec)) # random numbers between 0 and 1
    spk_poiss_idx = spk_poiss < pspike # boolean array of bins with spike
    spk_poiss_t = tvec[spk_poiss_idx] # use idxs to get corresponding spike time
    
    plt.vlines([spk_poiss_t, spk_poiss_t],-1, -0.5, Color = [0, 0, 0]) # note, plots all spikes in one command
    ax = plt.gca()
    ax.set_ylim([-1.5, 5])
    ax.set_xlim([0, 0.1])
    ax.set_yticks([])
    
    
    bin_size = 0.01 # in s
    max_t    = 1 # in s
    ac_poiss, bin_edges = acf(spk_poiss_t, bin_size, max_t)
    plt.figure()
    plt.plot(bin_edges[:-1]-bin_size/2, ac_poiss)
    plt.xlabel('lag [s]')
    plt.ylabel('acorr')
    
