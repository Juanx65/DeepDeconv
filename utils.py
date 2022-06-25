import numpy as np
import scipy.fft
from scipy.signal import find_peaks
from bisect import bisect_left
from collections import Counter


def FFT_roll(data, gauge, samp, v_ref):
    Nch, Nt = data.shape
    
    x = np.arange(Nch) * gauge
    x -= x.mean()
    freqs = scipy.fft.rfftfreq(Nt, d=1/samp)

    F = scipy.fft.rfft(data, axis=1)
    
    for i in range(Nch):
        F[i] = F[i] * np.exp(2j * np.pi * freqs * x[i] / v_ref)
    
    data_shift = np.real(scipy.fft.irfft(F, axis=1))
    
    return data_shift 


def peak_projection(data, prominence=3, distance=5):
    
    proj_t = np.max(data, axis=1)
    peaks_t, _ = find_peaks(proj_t, threshold=None, distance=distance, prominence=prominence)
    if len(peaks_t) == 0:
        return np.array([])
    
    peaks_v = np.zeros_like(peaks_t)
    for i, peak in enumerate(peaks_t):
        peaks_v[i] = np.argmax(data[peak])
    peaks = np.vstack([peaks_t, peaks_v]).T
    
    return peaks


def match_peaks(ts, targets, threshold):
    """ Match detections with manual picks """
    
    matches = Counter()
    dts = Counter()
    FPs = 0
    
    for i, t in enumerate(ts):
        ind = bisect_left(targets, t)
        
        if ind == len(targets):
            matches.append(ind - 1)
            continue
        
        dt_left = abs(t - targets[ind-1])
        dt_right = abs(t - targets[ind])
        
        if dt_left < dt_right:
            ind -= 1
            
        dt = min(dt_left, dt_right)
        
        if dt <= threshold:
            matches[targets[ind]] += 1
            dts[targets[ind]] += dt
        else:
            FPs += 1
        
    return matches, dts, FPs
