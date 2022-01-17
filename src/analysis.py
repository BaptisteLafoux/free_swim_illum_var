#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:48:57 2021

@author: baptistelafoux
"""

#%% Import modules 

import numpy as np 

import matplotlib.pyplot as plt 

from src.utilities import save_multi_image, set_suptitle, add_illuminance_on_plot, compute_focal_values, dataloader, center_bins, group_dataset, set_matplotlib_config

import palettable as pal 

import warnings; warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

global col; col = pal.tableau.TableauMedium_10.mpl_colors

#%% Plot functions     

def add_var_vs_light(ds, var, ax, correc=False, all_values=True, **kwargs) : 
    
    if all_values : ax.plot(ds.light, ds[var], '.', markersize=2, alpha=0.2, color=kwargs.get('color'))
    
    ds, ds_std = group_dataset(ds, 'light', n_bins = 32)
    
    if correc : 
        ii_dist_avg = ds.ii_dist.mean(dim=['neighbour', 'fish'])
        C = np.min(ii_dist_avg) /  ii_dist_avg

        ds[var] *= C**2

    p = ax.plot(ds.light, ds[var], 'o', **kwargs)
    ax.fill_between(ds.light, np.abs(ds[var]) - ds_std[var], np.abs(ds[var]) + ds_std[var], alpha=0.2, color=p[0].get_color())

def add_var_vs_time(ds, var, ax, downsmpl_per=180, add_illu=False, **kwargs):
        
    if add_illu: 
        add_illuminance_on_plot(ax, ds, scaling_factor=add_illu)
        ax.legend()
        
    win_size = int(downsmpl_per * ds.fps)
    ds_avg = ds.coarsen(time=win_size, boundary='trim').mean()
    
    ax.plot(ds.time, ds[var], 'k.', markersize=0.1, **kwargs)
    ax.plot(ds_avg.time, ds_avg[var], 'ko', **kwargs)
            
    ax.set_xlabel('Time [s]')
    
#### Plots vs time
    
def plot_pol_and_rot_param_vs_time(ds):
    
    fig, ax = plt.subplots(2, 1, sharex=True) 
    set_suptitle('Pol. & rot. param VS light', fig, ds)
    
    ds = np.abs(ds)
    add_var_vs_time(ds, 'pol_param', ax[0], add_illu=1); 
    add_var_vs_time(ds, 'rot_param', ax[1], add_illu=1); 
    
    ax[0].set_title('Pol. param $P$')
    ax[1].set_title('Rot. param $M$')

def plot_dist_vs_time(ds) : 
    
    fig, ax = plt.subplots(2, 1, sharex=True)
    set_suptitle('Distances vs time', fig, ds)
    
    ds = ds.mean(dim='fish', keep_attrs=True)
    add_var_vs_time(ds, 'nn_dist', ax[0], add_illu=2)

    ds = ds.mean(dim='neighbour', keep_attrs=True)
    add_var_vs_time(ds, 'ii_dist', ax[1], add_illu=15)
    
    ax[0].set_title('NN-D'); 
    ax[1].set_title('II-D')

def plot_vel_vs_time(ds) : 
    
    fig, ax = plt.subplots()
    set_suptitle('Vel. vs time', fig, ds)
    
    ds = ds.mean(dim='fish', keep_attrs=True)
    add_var_vs_time(ds, 'vel', ax, add_illu=2)
    ax.set_title('Velocity norm')
    
#### Plots vs light
        
def plot_pol_and_rot_param_vs_light(ds):
    
    fig, ax = plt.subplots()
    set_suptitle('Pol. & rot. param vs light', fig, ds)
    
    add_var_vs_light(np.abs(ds), 'pol_param', ax, correc=True, all_values=False, color=col[4], label='$P$')
    add_var_vs_light(np.abs(ds), 'rot_param', ax, correc=True, all_values=False, color=col[5], label='$M$')
    
    ax.legend()
    ax.set_xlabel('Illuminance [-]')
    ax.set_ylabel('[-]')
    
def plot_dist_vs_light(ds) : 
        
    fig, ax0 = plt.subplots(); ax1 = ax0.twinx()
    set_suptitle('Dist. vs light', fig, ds)
    
    ###
    ds = ds.mean(dim='fish', keep_attrs=True)
    add_var_vs_light(ds, 'nn_dist', ax0, color=col[1], linestyle='-', marker=None)

    ds = ds.mean(dim='neighbour', keep_attrs=True)
    add_var_vs_light(ds, 'ii_dist', ax1, color=col[2], linestyle='-', marker=None)
    
    ###
    ax0.set_ylabel('II-D [BL]', color=col[1])
    ax0.tick_params(axis='y', labelcolor=col[1])

    ax1.set_ylabel('NN-D [BL]', color=col[2])
    ax1.tick_params(axis='y', labelcolor=col[2])
    
    ax0.set_xlabel('Illuminance [-]')
    
def plot_vel_vs_light(ds) : 
    
    fig, ax = plt.subplots()
    set_suptitle('Vel. vs light', fig, ds)
    
    ds = ds.mean(dim='fish')
    add_var_vs_light(ds, 'vel', ax, color=col[3], linestyle='-', marker=None)
    
    ax.set_ylabel('Velocity norm $|u|$ [BL/s]')
    ax.set_xlabel('Illuminance [-]')
    
#### Focal data

def plot_focal_values(ds): 
    
    ds = compute_focal_values(ds)
    
    fig, ax = plt.subplots(1, 3)
    set_suptitle('Focal values', fig, ds)
    #### Presence density
    add_presence_density_focal(ds, ax[0], lim=4)

    
def add_presence_density_focal(ds, ax, lim, iso_contour_percentages=[0.1, 0.5]): # lim is in BL 
    
    percentages = np.sort(np.array(iso_contour_percentages))[::-1]
    
    x, y = ds.sr.to_numpy().reshape(-1, 2).T # these are the x, y postions of all the fish at all times, centered and rotated on each fish as focal (a lot of data) 
    valid_idx = ((np.abs(x) < lim) & (np.abs(y) < lim)) & (y!=0) & (x!=0) # remove all position further away than the lim and 0 position (positon of each fish relative to itself)
    
    presence_density, x_bins, y_bins = np.histogram2d(y[valid_idx], x[valid_idx], bins = 50)
    x_center, y_center = center_bins(x_bins, y_bins)
    
    presence_density /= np.sum(presence_density, axis=(0,1)) 
    
    #### 2D PDF map for location of neighbours around a focal fish 
    ax.contourf(x_center, y_center, presence_density, 6, cmap=plt.cm.autumn_r)
    
    #### Iso-contours plot
    # Compute integral of PDF to find levels that define region with an integral of *percentages*
    n = 100
    t = np.linspace(0, presence_density.max(), n)
    integral = ((presence_density >= t[:, None, None]) * presence_density).sum(axis=(1,2)) # compute integrals inside regions defined by *presence_density >= t) 
    
    
    f = interp1d(integral, t)
    levels = f(percentages)

    cs = ax.contour(x_center, y_center, gaussian_filter(presence_density, sigma=1.1), levels=levels, colors='k', linewidths = np.linspace(0.3, 1, len(percentages)))
    
    fmt = {}
    strs = ['{:d}%'.format(int(i * 100)) for i in percentages]
    
    for l, s in zip(cs.levels, strs):
        fmt[l] = s
        
    ax.clabel(cs, levels=levels, fmt=fmt, fontsize=8)
    ax.axis('scaled')
    


#%% Some wrappers 

def analysis_plot(ds, multi_analysis=False):
    '''
    A wrapper of all plotting function to execute one after the other. Comment those you don't want'

    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    plt.close('all')
    
    set_matplotlib_config()

    plot_pol_and_rot_param_vs_time(ds)    
    plot_dist_vs_time(ds)  
    plot_vel_vs_time(ds) 
        
    plot_pol_and_rot_param_vs_light(ds)
    plot_dist_vs_light(ds)
    plot_vel_vs_light(ds)

    plt.show()

    
def run_analysis(data_file_name):
    '''
    A wrapper

    Parameters
    ----------
    data_file_name : str
        .

    Returns
    -------
    None.

    '''
    dataset = dataloader(data_file_name)
    
    analysis_plot(dataset, multi_analysis=False) 
    save_multi_image(data_file_name)
    
    return dataset 
    
    
#%% Run file
    
if __name__ == "__main__":
    
    data_file_name = 'cleaned/3_VarLight/2022-01-06/1/trajectory.nc'
    
    
    
    _ = run_analysis(data_file_name)

    
