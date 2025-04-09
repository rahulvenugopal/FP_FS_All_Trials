# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 07:26:19 2025
- Visualise the Power and SLiding dynamics across trials

@author: Rahul Venugopal
"""

#%% Loading the libraries
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sp
import numpy as np
import mat73
import os

import mne
from matplotlib.colors import ListedColormap
from scipy.stats import spearmanr

#%% Load the data

power_data =  mat73.loadmat('tfpw_powers_rounded.mat')['rounded_power']
sliding_data =  mat73.loadmat('tfhz_sliders_rounded.mat')['rounded_sliding']

capacity_loads = sp.loadmat('wmload_sequence.mat')['capacities']

# Get channel names as a list
channel_names = sp.loadmat('chanlocs_55.mat')['chanlocs_55']
channel_names = [channel_names[0, i]['labels'][0] for i in range(channel_names.shape[1])]

colors = [(0.2706, 0.4588, 0.7059),
          (0.5686, 0.7490, 0.8588),
          (0.8784, 0.9529, 0.9725),
          (1.0000, 1.0000, 0.7490),
          (0.9961, 0.8784, 0.5647),
          (0.9882, 0.5529, 0.3490),
          (0.8431, 0.1882, 0.1529)]

cmap_becp = ListedColormap(colors)

eeg_bands = ['Theta', 'Alpha']

#%% Looping through

# subject_data_list: List of 3D arrays (channels x bands x trials)
# memory_list: List of 1D arrays (trials)

# Check data dimensions for consistency
num_subjects = len(capacity_loads[0,:])

results = []  # To collect correlation data

for subj_idx in range(num_subjects):
    subject_data = sliding_data[subj_idx]  # shape: (chans, bands, trials)
    memory_power = capacity_loads[0,subj_idx]  # shape: (trials,)

    channels, bands, trials = subject_data.shape

    for band in range(bands):
        for ch in range(channels):
            data = subject_data[ch, band, :]  # shape: (trials,)

            # Correlation between EEG and memory power across trials
            corr_coef, p_value = spearmanr(data, memory_power)

            # Append to results
            results.append({
                'Subject': subj_idx + 1,
                'Channel': channel_names[ch],
                'Band': eeg_bands[band],
                'Correlation': corr_coef,
                'P_Value': p_value
            })


            # # Plotting
            # plt.figure(figsize=(10, 4))
            # plt.plot(data, label='EEG Signal (per trial)')
            # plt.plot(memory_power, label='Memory Power')
            # plt.title(f'Subject {subj_idx+1} - Channel {ch+1}, Band {band+1}')
            # plt.xlabel('Trial')
            # plt.ylabel('Signal Value')
            # plt.legend()
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to CSV
df_results.to_csv('correlation_results_sliding.csv', index=False)

#%% Visualise the channel r and p value topos

# Define number of channels and known EEG channel names
n_channels = df_results['Channel'].nunique()

# Use a standard 10-20 montage or custom if needed
montage = mne.channels.make_standard_montage('standard_1020')

# Create a fake MNE Info object for topomap plotting
sfreq = 1  # dummy sampling freq
info = mne.create_info(ch_names=channel_names,
                       sfreq=sfreq, ch_types='eeg')
info.set_montage(montage)

# Create output folder
os.makedirs("topoplots_sliding", exist_ok=True)

# Loop over subjects and bands
subjects = df_results['Subject'].unique()
bands = df_results['Band'].unique()

for subj in subjects:
    for band in bands:
        df_sub = df_results[(df_results['Subject'] == subj) & (df_results['Band'] == band)]
        df_sub.reset_index(inplace = True)

        # Ensure ordering matches info channel order
        values_r = np.full(n_channels, np.nan)
        values_p = np.full(n_channels, np.nan)

        for index, row in df_sub.iterrows():
            ch_idx = row['Channel']
            values_r[index] = row['Correlation']
            values_p[index] = row['P_Value']

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        im,cm = mne.viz.plot_topomap(values_r, info, axes=axes[0],
                             show=False, names=channel_names,
                             cmap=cmap_becp,
                             vlim=(min(values_r),max(values_r)))
        
            
        axes[0].set_title(f'Subject {subj} - Band {band}\nCorrelation (r)')

        # ax_x_start = 0.925
        # ax_x_width = 0.01
        # ax_y_start = 0.1
        # ax_y_height = 0.8

        # cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im) # put a comma and add cbar_ax

        im1, cm1 = mne.viz.plot_topomap(values_p, info, axes=axes[1],
                             show=False, names=None,
                             cmap=cmap_becp,
                             vlim=(0,1))
        axes[1].set_title(f'Subject {subj} - Band {band}\nP-Values')

        clb = fig.colorbar(im1)

        plt.tight_layout()

        plt.savefig(f'topoplots_sliding/subj{subj}_band{band}.png', dpi=600)
        plt.close()

print("All topoplots saved in 'topoplots_sliding/' folder.")

#%% Line traces of working memory capacity and power/sliding
