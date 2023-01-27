#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 16:48:57 2021

@author: baptistelafoux
"""

#%% Import modules

import numpy as np
import matplotlib.pyplot as plt

from utils.graphic import save_multi_image, set_suptitle, add_illuminance_on_plot, set_matplotlib_config
from utils.loader import dataloader
from utils.data_operations import group_dataset, add_modified_rot_param

import warnings
warnings.filterwarnings("ignore")


#%% Plot functions

def add_var_vs_light(ds, var, ax, n_samples=12, correc=False, all_values=True, **kwargs):
    '''
    A function to plot any variable from a Dataset (var) on a given Axe (ax) with respect to light level

    Parameters
    ----------
    ds : Dataset 
        .
    var : str
        A variable name.
    ax : AxesSubplot
        .
    correc : bool, optional
        Correction on data based on mean II-D. The default is False.
    all_values : TYPE, optional
        Plot all values as small points (can get crowded). The default is True.
    **kwargs : kwargs
        Optional argument for ax.plot.

    Returns
    -------
    None.

    '''

    if all_values:
        ax.plot(ds.light, ds[var], '.', markersize=1, alpha=0.05, color=kwargs.get('color'))

    if correc:
        ii_dist_avg = ds.ii_dist.mean(dim=['neighbour', 'fish'])
        C = 1 - (ii_dist_avg - np.min(ii_dist_avg)) / np.max(ii_dist_avg)

        ds[var] *= C
        
    ds, ds_std = group_dataset(ds, 'light', n_bins=52)
   

    p = ax.plot(ds.light, ds[var], 'o', **kwargs)
    ax.fill_between(ds.light, np.abs(ds[var]) - 1.96 * ds_std[var] / np.sqrt(n_samples), np.abs(ds[var]) +
                    1.96 * ds_std[var] / np.sqrt(n_samples), alpha=0.2, color=p[0].get_color(), edgecolor='none')

    return ds
     


def add_var_vs_time(ds, var, ax, downsmpl_per=180, add_illu=False, **kwargs):
    '''
    A function to plot any variable from a Dataset (var) on a given Axe (ax) with respect to time

    Parameters
    ----------
    ds : Dataset 
        .
    var : str
        A variable name.
    ax : AxesSubplot
        .
    downsmpl_per : int, optional
        In seconds, downsampling period for plotting understandable curves. The default is 180.
    add_illu : bool, optional
        True if you want to plot the illuminance signal in your AxesSubplot too. The default is False.
    **kwargs : kwargs
        Optional argument for ax.plot.

    '''

    if add_illu:
        add_illuminance_on_plot(ax, ds, scaling_factor=add_illu)
        ax.legend()

    win_size = int(downsmpl_per * ds.fps)
    ds_avg = ds.coarsen(time=win_size, boundary='trim').mean()

    p = ax.plot(ds_avg.time, ds_avg[var], **kwargs)
    ax.plot(ds.time, ds[var], '.', color=p[0].get_color(), alpha=0.2, mew=0)

    ax.set_xlabel('Time [s]')

#### Plots vs time


def plot_pol_and_rot_param_vs_time(ds):

    fig, ax = plt.subplots(2, 1, sharex=True)
    set_suptitle('Pol. & rot. param VS light', fig, ds)

    ds = np.abs(ds)
    add_var_vs_time(ds, 'pol_param', ax[0], add_illu=1)
    add_var_vs_time(ds, 'rot_param', ax[1], add_illu=1)

    ax[0].set_title('Pol. param $P$')
    ax[1].set_title('Rot. param $M$')


def plot_dist_vs_time(ds):

    fig, ax = plt.subplots(2, 1, sharex=True)
    set_suptitle('Distances vs time', fig, ds)

    ds = ds.mean(dim='fish', keep_attrs=True)
    add_var_vs_time(ds, 'nn_dist', ax[0], add_illu=2)

    ds = ds.mean(dim='neighbour', keep_attrs=True)
    add_var_vs_time(ds, 'ii_dist', ax[1], add_illu=15)

    ax[0].set_title('NN-D')
    ax[1].set_title('II-D')


def plot_vel_vs_time(ds):

    fig, ax = plt.subplots()
    set_suptitle('Vel. vs time', fig, ds)

    ds = ds.mean(dim='fish', keep_attrs=True)
    add_var_vs_time(ds, 'vel', ax, add_illu=2)
    ax.set_title('Velocity norm')

#### Plots vs light


def plot_pol_and_rot_param_vs_light(ds):

    fig, ax = plt.subplots()
    set_suptitle('Pol. & rot. param vs light', fig, ds)

    add_var_vs_light(np.abs(ds), 'pol_param', ax, correc=True, all_values=False, color='C4', label='$P$')
    add_var_vs_light(np.abs(ds), 'rot_param', ax, correc=True, all_values=False, color='C6', label='$M$')

    ax.legend()
    ax.set_xlabel('Illuminance [-]')
    ax.set_ylabel('[-]')


def plot_dist_vs_light(ds):

    fig, ax0 = plt.subplots()
    ax1 = ax0.twinx()
    set_suptitle('Dist. vs light', fig, ds)

    ###
    ds = ds.mean(dim='fish', keep_attrs=True)
    add_var_vs_light(ds, 'nn_dist', ax0, color='C1', linestyle='-', marker=None)

    ds = ds.mean(dim='neighbour', keep_attrs=True)
    add_var_vs_light(ds, 'ii_dist', ax1, color='C2', linestyle='-', marker=None)

    ###
    ax0.set_ylabel('II-D [BL]', color='C1')
    ax0.tick_params(axis='y', labelcolor='C1')

    ax1.set_ylabel('NN-D [BL]', color='C2')
    ax1.tick_params(axis='y', labelcolor='C2')

    ax0.set_xlabel('Illuminance [-]')


def plot_vel_vs_light(ds):

    fig, ax = plt.subplots()
    set_suptitle('Vel. vs light', fig, ds)

    ds = ds.mean(dim='fish')
    add_var_vs_light(ds, 'vel', ax, color='C3', linestyle='-', marker=None)

    ax.set_ylabel('Velocity norm $|u|$ [BL/s]')
    ax.set_xlabel('Illuminance [-]')


def figure_density_centered_on_centroid(ds, n_bins=5, bin_type='log', histo_n_bins=40): 
    
    from utils.graphic import center_bins
    from scipy.ndimage import gaussian_filter
    
    s_centered_on_centroid = ds.s - ds.center_of_mass
    
    if bin_type == 'log': bins = np.logspace(-1.5, 0, n_bins+1)
    if bin_type == 'lin': bins = np.linspace(0, 1, n_bins+1)
    
    fig, axs = plt.subplots(1, n_bins, figsize=(25, 5), sharey=True)
    fig.set_tight_layout(True)

    print(f"\nPlotting presence density with respect to light ({n_bins} different levels with {bin_type}-type repartition)")
    for i, (l_l, l_r) in enumerate(zip(bins[:-1], bins[1:])):
    
        x, y = s_centered_on_centroid.where((ds.light > l_l) & (ds.light < l_r), drop=True).to_numpy().reshape((-1, 2)).T
    
        non_nan_idx = ~np.isnan(x) & ~np.isnan(y)
        x = x[non_nan_idx]; y = y[non_nan_idx]
        
        print(f'{i+1}/{n_bins} - E = [{l_l:.2f} - {l_r:.2f}] - Number of positions kept : {np.sum(non_nan_idx):.1e}')
        
        density_map, y_edges, x_edges = np.histogram2d(y, x, bins=histo_n_bins, density=True)
    
        x_center, y_center = center_bins(x_edges, y_edges)
    
        axs[i].pcolormesh(x_center, y_center, gaussian_filter(density_map, 2), cmap='Reds', shading='auto')
    
        axs[i].set_title(rf'$\bar E$ = {l_l:.2f} - {l_r:.2f}')
        axs[i].axis('scaled')
        
        if 'experiment' in ds.dims : tank_size = np.array(ds.tank_size).max(axis=0)
        else : tank_size = ds.tank_size
        axs[i].axis(np.c_[-tank_size/2, tank_size/2].flatten())

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
