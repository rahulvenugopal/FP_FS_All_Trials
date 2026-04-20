# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 22:15:13 2025
- Find consistent channels in EEG which are modulated by WM capacity
- This is checked across subjects as a heatmap

@author: Rahul Venugopal
"""
#%% Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne

from matplotlib.colors import ListedColormap

#%% Color palette
colors = [(0.2706, 0.4588, 0.7059),
          (0.5686, 0.7490, 0.8588),
          (0.8784, 0.9529, 0.9725),
          (1.0000, 1.0000, 0.7490),
          (0.9961, 0.8784, 0.5647),
          (0.9882, 0.5529, 0.3490),
          (0.8431, 0.1882, 0.1529)]

cmap_becp = ListedColormap(colors)

# Load the data
df = pd.read_csv('correlation_csvs/correlation_aperiodic_exponent.csv')

# Choose a frequency band to visualize
band_to_plot = 'Aperiodic_Exponent'
df_band = df[df['Band'] == band_to_plot]

# Count how many subjects had significant p-values per channel
significance_df = df_band[df_band['P_Value'] < 0.05]
subject_counts = df_band.groupby('Channel')['Subject'].nunique()
significant_counts = significance_df.groupby('Channel')['Subject'].nunique()

# Compute proportion of significant subjects per channel
proportions = (significant_counts / subject_counts).fillna(0)

# Prepare for topomap
channel_names = proportions.index.tolist()
values = proportions.values

# Create MNE info with only those channels
info = mne.create_info(ch_names=channel_names, sfreq=1000, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# Map data to correct channel positions
data = []
ordered_ch_names = info.ch_names
for ch in ordered_ch_names:
    if ch in proportions:
        data.append(proportions[ch])
    else:
        data.append(np.nan)  # Fill missing channels with NaN

#%% Visualise the topo

# # Create a mask to place white dots (mask is True at every electrode location)
# mask = np.ones(len(data), dtype=bool)

# # Define mask parameters to plot white dots (no outline)
# mask_params = dict(marker='o', markerfacecolor='white', markersize=4, markeredgewidth=0, linewidth=0)

# Plot topomap
fig, ax = plt.subplots(figsize=(8, 6))
im, cm = mne.viz.plot_topomap(
    np.array(data), info,
    show=True,
    contours=0,
    # mask=mask,
    # mask_params=mask_params,
    axes=ax,
    names=ordered_ch_names,
    cmap=cmap_becp)

# Title
ax.set_title(f'Topomap of percentage of significant correlations (p < 0.05) - {band_to_plot} Band')

fig.colorbar(im, ax=ax).set_label('% of subjects who showed modulation by working memory capacity', fontsize=10)

# Layout and save
plt.tight_layout()
plt.savefig(f'PercentSignificant_band{band_to_plot}.png', dpi=600)
plt.close()