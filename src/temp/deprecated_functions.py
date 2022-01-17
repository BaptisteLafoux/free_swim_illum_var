#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 15:35:30 2022

@author: baptistelafoux
"""

import src.analysis 

def plot_polarization_and_rotation_param(ds, ax, downsampling_period_seconds=2*int(60), filt_order=3):
    
    q = int(downsampling_period_seconds * np.mean(ds.fps, dtype=int))
    
    #### Plotting pol. param
    ax[0].set_title('Polarization param. ($T_{avg}$ = ' + str(datetime.timedelta(seconds=downsampling_period_seconds)) + ')')
    ax[0].plot(ds.time, ds.pol_param.T, 'k.', markersize=0.1)
    
    add_illuminance_on_plot(ax[0], ds)
    
    # downsampling
    downsampled_pol_param = decimate(np.abs(ds.pol_param), q=q, n=filt_order)
    downsampled_time = np.interp(np.linspace(0, ds.time.max(), len(downsampled_pol_param)), ds.time, ds.time)
    ax[0].plot(downsampled_time, downsampled_pol_param, 'ko')
    ax[0].legend()
    ax[0].set_xlabel('Time [s]')
    
    #### Plotting rot. param
    ax[1].set_title('Rotation param. ($T_{avg}$ = ' + str(datetime.timedelta(seconds=downsampling_period_seconds)) + ')')
    ax[1].plot(ds.time, np.abs(ds.rot_param), 'k.', markersize=0.1)
    add_illuminance_on_plot(ax[1], ds)
    
    # downsampling
    downsampled_rot_param = decimate(np.abs(ds.rot_param), q=q, n=filt_order)
    downsampled_time = np.interp(np.linspace(0, ds.time.max(), len(downsampled_rot_param)), ds.time, ds.time)
    ax[1].plot(downsampled_time, downsampled_rot_param, 'ko')
    ax[1].legend()
    ax[1].set_xlabel('Time [s]')
    
    plt.tight_layout()
    
def plot_pol_and_rot_param_VS_light(ds, downsampling_period_seconds=3*int(60), filt_order=3):
    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    set_suptitle('$P$, $M$ = f(Illuminance)', fig, ds)
    
    
    #### With decimate method to downsample signals 
    q = int(downsampling_period_seconds * np.mean(ds.fps, dtype=int))
    
    ax[0].set_title('Polarization $P$')
    ax[0].plot(ds.light, ds.pol_param, 'k.', markersize=0.1)
    ax[0].plot(decimate(ds.light, q=q, n=filt_order), decimate(ds.pol_param, q=q, n=filt_order), 'ko')
    ax[0].set_xlabel('Illuminance []') 
    ax[0].set_ylabel('$P$ []') 
    
    ax[1].set_title('Rotation $M$')
    ax[1].plot(ds.light, np.abs(ds.rot_param), 'k.', markersize=0.1)
    ax[1].plot(decimate(ds.light, q=q, n=filt_order), decimate(np.abs(ds.rot_param), q=q, n=filt_order), 'ko')
    ax[1].set_xlabel('Illuminance []')
    ax[1].set_ylabel('$M$ []') 
    
    #### With groupby method from xarrays 
    n_bins = 52
    ds_group_avg = ds.groupby_bins('light', bins=n_bins, precision=5).mean()
    ds_group_std = ds.groupby_bins('light', bins=n_bins, precision=5).std()
    
    ii_dist_avg = ds_group_avg.ii_dist.mean(dim=['neighbour', 'fish'])
    #C = (np.min(ii_dist_avg) / ii_dist_avg)**2
    C = 1
    
    ds_group_avg['rot_param'] *= C; ds_group_avg['pol_param'] *= C
    
    ax[2].plot(ds_group_avg.light, np.abs(ds_group_avg.rot_param), 'o', label='Rot. param.')
    ax[2].fill_between(ds_group_avg.light, np.abs(ds_group_avg.rot_param) - ds_group_std.rot_param, np.abs(ds_group_avg.rot_param) + ds_group_std.rot_param, alpha=0.2)
    
    ax[2].plot(ds_group_avg.light, ds_group_avg.pol_param, 'o', label='Pol. param.')
    ax[2].fill_between(ds_group_avg.light, np.abs(ds_group_avg.pol_param) - ds_group_std.pol_param, np.abs(ds_group_avg.pol_param) + ds_group_std.pol_param, alpha=0.2)
    
    ax[2].legend(loc=4)
    ax[2].set_xlabel('Illuminance []')
        
    plt.tight_layout()
    
    
def plot_dist(ds): 
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    
    set_suptitle('Distances', fig, ds)
    
    #### Avg interindividual distance
    avg_ii_dist = ds.ii_dist.mean(dim=['fish', 'neighbour'])
    ax[0].plot(ds.time, avg_ii_dist, label='IID')
    add_illuminance_on_plot(ax[0], ds, scaling_factor=20)
    ax[0].legend()
    ax[0].set_xlabel('Time [s]'); ax[0].set_ylabel('[BL]')
    
    #### Avg nearest-neigbour distance
    
    
    avg_nn_dist = ds.nn_dist.mean(dim='fish')
    ax[1].plot(ds.time, avg_nn_dist, label='NND')
    add_illuminance_on_plot(ax[1], ds, scaling_factor=3)
    ax[1].legend()
    ax[1].set_xlabel('Time [s]'); ax[1].set_ylabel('[BL]')
    
    plt.tight_layout()
    
#### Velocity norm 
    
def plot_vel(ds, downsampling_period_seconds=3*int(60), filt_order=3) :
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    set_suptitle('Velocity norm', fig, ds)
    q = int(downsampling_period_seconds * np.mean(ds.fps, dtype=int))
    
    vel = xarray_vector_norm(ds.v, dim='space').mean(dim='fish')
    
    #### Velocity norm relative to time 
    ax[0].plot(ds.time, vel, label='$|u|$')
    add_illuminance_on_plot(ax[0], ds)
    ax[0].legend()
    ax[0].set_xlabel('Time [s]')             
    ax[0].set_ylabel('[BL/s]')
    
    #### Velocity norm relative to light
    
    ax[1].plot(ds.light, vel, 'k.', markersize=0.1)
    ax[1].plot(decimate(ds.light, q=q, n=filt_order), decimate(vel, q=q, n=filt_order), 'ko')
    ax[1].set_xlabel('Illuminance []')
    ax[1].set_ylabel('$|u|$ [BL/s]') 
    
    plt.tight_layout()