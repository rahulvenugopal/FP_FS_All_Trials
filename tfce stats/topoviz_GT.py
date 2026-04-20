# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 15:28:38 2023
- We load the 5D matrix for all five GT params
- Trimmed mean across subjects
- Loop through params and bands
@author: Rahul Venugopal
"""
#%% Loading libraries
import scipy.io as sp
import numpy as np
import mne
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats

#%% Load the metadata which tells us the trial numbers
GT_all = sp.loadmat('results/pernode_GT.mat')['graphdata']

features = ['clustering', 'global_efficiency', 'centrality',
            'participation_coefficient','degree']
# For title
features_titles = ['Clustering coefficient', 'Global efficiency', 'Centrality',
            'Participation coefficient','Degree']

# 7 level colormap
colors = [(0.2706, 0.4588, 0.7059),
          (0.5686, 0.7490, 0.8588),
          (0.8784, 0.9529, 0.9725),
          (1.0000, 1.0000, 0.7490),
          (0.9961, 0.8784, 0.5647),
          (0.9882, 0.5529, 0.3490),
          (0.8431, 0.1882, 0.1529)]

cmap_becp = ListedColormap(colors)

# create an mne info
# Sampling rate and channel types
sfreq = 1000  # Sampling rate in Hz
ch_types = ['eeg'] * 55
eeg_chans = ['Fp1','Fz','F3','F7','FC5','FC1','C3','T7','CP5','CP1','Pz','P3','P7',
             'O1','Oz','O2','P4','P8','CP6','CP2','Cz','C4','T8','FC6','FC2','F4',
             'F8','Fp2','AF7','AFz','F1','F5','FT7','FC3','C1','C5','TP7','P1',
             'P5','PO7','PO3','POz','PO4','PO8','P6','P2','CPz','TP8','C6','C2',
             'FC4','FT8','F6','AF8','F2']

# Create mne.Info object
info = mne.create_info(eeg_chans, sfreq, ch_types)

info.set_montage('standard_1005')

#%% Loop through parameters and thresholds

# we can plot for all thresholds as well
for threshold in range(28,33):
    
    for params in range(np.shape(GT_all)[0]):
        # Pick the GT parameter
        # Trimmed mean of prewm subjects only at that threshold
        topo_data = stats.trim_mean(GT_all[params,1][:,:,1::2,threshold,:], 0.1,2)
    
        # creating a list of plots in specific order to be picked up from loop
        plot_data = [topo_data[0,0,:],topo_data[1,0,:],topo_data[2,0,:],topo_data[3,0,:],
                     topo_data[4,0,:],topo_data[5,0,:],topo_data[6,0,:],
    
                     topo_data[0,1,:],topo_data[1,1,:],topo_data[2,1,:],topo_data[3,1,:],
                     topo_data[4,1,:],topo_data[5,1,:],topo_data[6,1,:],
                     
                     topo_data[0,2,:],topo_data[1,2,:],topo_data[2,2,:],topo_data[3,2,:],
                     topo_data[4,2,:],topo_data[5,2,:],topo_data[6,2,:],
                     
                     topo_data[0,3,:],topo_data[1,3,:],topo_data[2,3,:],topo_data[3,3,:],
                     topo_data[4,3,:],topo_data[5,3,:],topo_data[6,3,:]]
    
        # Create a figure with 3*5 subplots in three parts (error of ytick and numpy array)
        fig, axes = plt.subplots(nrows=4, ncols=7)
    
        # Adjust the subplot parameters
        plt.subplots_adjust(left=0.03, right=0.94, bottom=0.01, top=0.98)
    
        # set figure size
        plt.rcParams['figure.figsize'] = [80, 20]
    
        row_title = ['SS 2', 'SS 3', 'SS 4', 'SS 5', 'SS 6', 'SS 7', 'SS 8']
    
        column_title = ['Theta', 'Alpha', 'Beta', 'Gamma']
    
        # Setting up a row header
        for ax, row in zip(axes[:,0], column_title):
            ax.set_ylabel(row, rotation=90, fontsize = 18, fontweight='bold')
    
        # Setting up the location in grid
        for plot_no in range(28):
            if plot_no < 7:
                row_id = 0
                col_id = plot_no
            elif plot_no < 14:
                row_id = 1
                col_id = plot_no - 7
            elif plot_no < 21:
                row_id = 2
                col_id = plot_no - 14
            elif plot_no < 28:
                row_id = 3
                col_id = plot_no - 21
                
            # Setting up the min and max for colorbars
            ref_min = np.min(np.concatenate(plot_data))
            ref_max = np.max(np.concatenate(plot_data))
    
            # Let MNE do the topomap (hey, why don't you add a colorbar!)
            im,cm = mne.viz.plot_topomap(plot_data[plot_no],
                                 info,
                                 vlim = (ref_min, ref_max),
                                 cmap = cmap_becp,
                                 axes = axes[row_id,col_id])
            if plot_no < 7:
                axes[row_id,col_id].set_title(row_title[plot_no],
                                              fontsize=18, fontweight='bold')
    
        ax_x_start = 0.96
        ax_x_width = 0.01
        ax_y_start = 0.1
        ax_y_height = 0.8
        
        # plt.suptitle(features_titles[params],
        #              fontsize=40, fontweight='bold')
        
        # Add some whitespaces above the plots
        #plt.subplots_adjust(top=0.85)
    
        cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im, cax=cbar_ax)
        # clb.ax.set_title('Frequency (Hz)',fontsize=8) # title on top of colorbar
    
        save_path_name = 'results/setsizes_prewm/topos_' + str(threshold) + '_' + features[params] +'_.png'
    
        plt.savefig(save_path_name, dpi = 600)
        plt.close()

#%% Make another round of topomaps based on optimal setsize | Below, Opt, Above

# We are picking all the PREWM files now
# subtracting 3 from optsetsize so that we can use that as an index
opt_setsize = np.squeeze(sp.loadmat('results/opt_setsize_all.mat')['Opt_array_all'] - 2)[1::2]

# postwm
opt_setsize = np.squeeze(sp.loadmat('results/opt_setsize_all.mat')['Opt_array_all'] - 2)[0::2]

GT_all = sp.loadmat('results/pernode_GT.mat')['graphdata']

for threshold in range(28,33):
    
    for params in range(np.shape(GT_all)[0]):
        # Pick the GT parameter
        # Trimmed mean of prewm subjects only at that threshold
        # Pick only the optimised
        data = GT_all[params,1][:,:,1::2,threshold,:]
        # we need to create three datasets now which hold the three sets based on opt
        data_below = []
        data_opt = []
        data_above = []
        
        for indices,values in enumerate(opt_setsize):            
            data_below.append(np.nanmean(data[0:values,:,indices,:], axis=0))
            data_opt.append(data[values,:,indices,:])
            data_above.append(np.nanmean(data[values+1:7,:,indices,:], axis=0))
        
        # Creating a 4D array from 3 lists each having 36 subjects
        stacked_list1 = np.stack(data_below, axis=0)
        stacked_list2 = np.stack(data_opt, axis=0)
        stacked_list3 = np.stack(data_above, axis=0)
        
        topo_data = stats.trim_mean(np.stack([stacked_list1, stacked_list2, stacked_list3], axis=0), 0.1,1)
    
        # creating a list of plots in specific order to be picked up from loop
        plot_data = [topo_data[0,0,:],topo_data[1,0,:],topo_data[2,0,:],
    
                     topo_data[0,1,:],topo_data[1,1,:],topo_data[2,1,:],
                     
                     topo_data[0,2,:],topo_data[1,2,:],topo_data[2,2,:],
                     
                     topo_data[0,3,:],topo_data[1,3,:],topo_data[2,3,:]]
    
        # Create a figure with 3*5 subplots in three parts (error of ytick and numpy array)
        fig, axes = plt.subplots(nrows=4, ncols=3)
    
        # Adjust the subplot parameters
        plt.subplots_adjust(left=0.03, right=0.94, bottom=0.01, top=0.98)
    
        # set figure size
        plt.rcParams['figure.figsize'] = [15, 15]
    
        row_title = ['Below', 'Optimal', 'Above']
    
        column_title = ['Theta', 'Alpha', 'Beta', 'Gamma']
    
        # Setting up a row header
        for ax, row in zip(axes[:,0], column_title):
            ax.set_ylabel(row, rotation=90, fontsize = 18, fontweight='bold')
    
        # Setting up the location in grid
        for plot_no in range(12):
            if plot_no < 3:
                row_id = 0
                col_id = plot_no
            elif plot_no < 6:
                row_id = 1
                col_id = plot_no - 3
            elif plot_no < 9:
                row_id = 2
                col_id = plot_no - 6
            elif plot_no < 12:
                row_id = 3
                col_id = plot_no - 9
                
            # Setting up the min and max for colorbars
            ref_min = np.min(np.concatenate(plot_data))
            ref_max = np.max(np.concatenate(plot_data))
    
            # Let MNE do the topomap (hey, why don't you add a colorbar!)
            im,cm = mne.viz.plot_topomap(plot_data[plot_no],
                                 info,
                                 vlim = (ref_min, ref_max),
                                 cmap = cmap_becp,
                                 axes = axes[row_id,col_id])
            if plot_no < 3:
                axes[row_id,col_id].set_title(row_title[plot_no],
                                              fontsize=18, fontweight='bold')
    
        ax_x_start = 0.96
        ax_x_width = 0.01
        ax_y_start = 0.1
        ax_y_height = 0.8
        
        # plt.suptitle(features_titles[params],
        #              fontsize=40, fontweight='bold')
        
        # Add some whitespaces above the plots
        #plt.subplots_adjust(top=0.85)
    
        cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im, cax=cbar_ax)
        plt.tight_layout()
        # clb.ax.set_title('Frequency (Hz)',fontsize=8) # title on top of colorbar
    
        save_path_name = 'results/opt_results_prewm/topos_' + str(threshold+1) + '_' + features[params] +'_.png'
    
        plt.savefig(save_path_name, dpi = 600)
        plt.close()
        
#%% Create a master array for just PREWM files
GT_all = sp.loadmat('results/pernode_GT.mat')['graphdata']
opt_setsize = np.squeeze(sp.loadmat('results/opt_setsize_all.mat')['Opt_array_all'] - 2)[1::2]

# First flatten the parameter from the cell
# We are looping through the params, thresholds, bands, subjects
list_of_params = []
for params in range(np.shape(GT_all)[0]):
    data = GT_all[params,1]
    list_of_params.append(data)
    
master_array = np.stack(list_of_params, axis=0)            

# drop all the postwm files
prewm_array = master_array[:,:,:,1::2,:,:]

# Get Below, Opt and Above datasets

# we need to create three datasets now which hold the three sets based on opt
data_below = []
data_opt = []
data_above = []

for indices,values in enumerate(opt_setsize):            
    data_below.append(np.nanmean(prewm_array[:,0:values,:,indices,:,:], axis=1))
    data_opt.append(prewm_array[:,values,:,indices,:,:])
    data_above.append(np.nanmean(prewm_array[:,values+1 : 7,:,indices,:,:], axis=1))

# Creating a 4D array from 3 lists each having 36 subjects
below_array = np.stack(data_below, axis=0)
opt_array = np.stack(data_opt, axis=0)
above_array = np.stack(data_above, axis=0)

master_array_prewm = np.stack((below_array,opt_array,above_array), axis=0)

sp.savemat('/serverdata/ccshome/rahul/Desktop/FC_GT_ACDMT/results/master_array_prewm.mat',
           {'master_array_prewm': master_array_prewm})













