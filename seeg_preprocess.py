import numpy as np
from scipy.signal import butter, filtfilt, hilbert, decimate
import pandas as pd
import glob
import os

def butter_filter(data, low_freq, high_freq, fs, btype):
    nyquist = 0.5 * fs
    
    if low_freq:
        low = low_freq / nyquist
    else:
        low = None
        
    if high_freq:
        high = high_freq / nyquist
    else:
        high = None
    
    # Adjust the filter parameters based on the filter type
    if btype == 'low':
        wn = high
    elif btype == 'high':
        wn = low
    elif btype == 'band':
        wn = [low, high]
    else:
        raise ValueError(f"Invalid filter parameters for {btype} filter.")
    
    b, a = butter(4, wn, btype=btype)
    return filtfilt(b, a, data, axis=0)




def filter_signal(eeg, fs, ds_factor=5):
    # Band-pass filter the data
    eeg_fs = butter_filter(eeg, 0.5, 150, fs, 'band')
    # Compute analytic amplitude
    filtered_signal = np.abs(hilbert(eeg_fs, axis=0))
    # Downsample
    filtered_signal = decimate(filtered_signal, ds_factor, axis=0) # from 1kHz to 200Hz
    return filtered_signal

def normalize_channel_data(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def process_eeg(eeg):
    

    filtered_signal = filter_signal(eeg, 1000,5)
    # Data normalization
    normalized_signal = normalize_channel_data(filtered_signal)
    

    if not os.path.exists(path_output):
        os.makedirs(path_output)
        
    np.save(os.path.join(path_output,f'{fname}_seeg.npy'), normalized_signal)



if __name__=="__main__":


    path_output = '/path/to/output'
    
    extension = 'csv'
    directory = '/path/to/input'
    all_filenames = sorted(glob.glob('{}/*.{}'.format(directory, extension)))
    

    for filename in all_filenames:
        
        filename_split = filename.split('\\')
        fname = filename_split[-1]
        fname = fname.split('.')[0]
        eeg = pd.read_csv(filename).values
        eeg = eeg[:,1:]
        eeg = eeg.T

        process_eeg(eeg)