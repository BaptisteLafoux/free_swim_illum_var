#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:33:12 2022

Plot figures for the article
@author: baptistelafoux
"""

#%% Import modules

from utils.loader import dataloader, dataloader_multiple
from utils.graphic import set_matplotlib_config, add_illuminance_on_plot, plot_frame_with_trajectories
from utils.data_operations import add_modified_rot_param
from src.analysis import add_var_vs_light, add_var_vs_time, figure_density_centered_on_centroid
import matplotlib.pyplot as plt

import glob

#%% Plot functions


def fig_timeserie_graph(ds, downsampling_period=10):

    ii_dist_avg = ds.ii_dist.mean(dim=['neighbour', 'fish'])
    C = 1 - (ii_dist_avg - ii_dist_avg.min())/ii_dist_avg.max()

    #ds['rot_param'] = ds.rot_param * C
    
    #ds = add_modified_rot_param(ds)

    ###
    fig, ax = plt.subplots(figsize=(3.45, 2.5))

    add_illuminance_on_plot(ax, ds, scaling_factor=1,
                            label=r'$\bar{E}$', color='w', linewidth=1, linestyle='--')

    ds_cors = ds.coarsen(time=int(downsampling_period*ds.fps), boundary='trim').mean()

    # warning here : downsampling twice, so real downsampling period is weird
    add_var_vs_time(abs(ds_cors), 'rot_param', ax, downsmpl_per=2, color='C3', label='$\mathcal{M}$')
    add_var_vs_time(abs(ds_cors), 'pol_param', ax, downsmpl_per=2, color='C4', label='$\mathcal{P}$')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('[-]')
    ax.legend()

    if SAVE_FIG:
        fig.savefig('output/subfigures/fig_timeserie_graph_jmc.pdf', transparent=True, format='pdf')


def fig_timeserie_snapshots(ds):

    times = [359, 800, 1350, 2320, 3197]

    ###
    for i, t in enumerate(times):

        fig, ax = plt.subplots(figsize=(3, 4))
        plot_frame_with_trajectories(ds, ax, t=t, color='k', noBG=True)
        
        print(rf'$\bar E$ = {ds.sel(time=t).light.data:.2f}')

        if SAVE_FIG:
            fig.savefig(f'output/subfigures/snap_{i}.pdf', transparent=True,
                        format='pdf', bbox_inches='tight', pad_inches=0)


def fig_general_plot(ds):

    fig, ax = plt.subplots(2, 1, figsize=(3.45, 4.5), sharex=True)

    ###
    add_var_vs_light(abs(ds), 'pol_param', ax[0], correc=True, all_values=False,
                     color='C4', label='Polarization $\mathcal{P}$', marker=None, linestyle='-')
    add_var_vs_light(abs(ds), 'rot_param', ax[0], correc=True, all_values=False,
                     color='C3', label='Milling $\mathcal{M}$', marker=None, linestyle='-')

    ax0 = ax[1]
    ax1 = ax0.twinx()

    ds = ds.mean(dim='fish', keep_attrs=True)
    add_var_vs_light(ds, 'nn_dist', ax[1], color='C1', linestyle='-', marker=None, all_values=False)

    ds = ds.mean(dim='neighbour', keep_attrs=True)
    add_var_vs_light(ds, 'ii_dist', ax1, color='C2', linestyle='-', marker=None, all_values=False)

    ###
    ax0.set_ylabel('NN-D [BL]', color='C1')
    ax0.tick_params(axis='y', labelcolor='C1')

    ax1.set_ylabel('II-D [BL]', color='C2')
    ax1.tick_params(axis='y', labelcolor='C2')

    ax[0].set_ylim([0, 1])
    ax[0].legend()
    ax[1].set_xlabel(r'Normalized illuminance $\bar E$ [-]')
    ax[0].set_ylabel('[-]')

    if SAVE_FIG:
        fig.savefig('output/fig_general_dark.pdf', transparent=True, format='pdf')


def fig_example_frame(ds, t=3400):

    ##
    fig, ax = plt.subplots()

    plot_frame_with_trajectories(ds, ax, t=t, tr_len=0, color='none', noBG=False)
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)

    if SAVE_FIG:
        fig.savefig('output/subfigures/frame_example_bg.png', dpi=300)

    ##
    fig, ax = plt.subplots()
    plot_frame_with_trajectories(ds, ax, t=t, tr_len=0, color='C3', noBG=True)

    ds *= ds.mean_BL_pxl

    t_plot = t - 1/ds.fps
    x, y = ds.s.sel(time=slice(t_plot-1, t_plot)).to_numpy().reshape((-1, 2)).T

    ax.plot(x, y, 'k.', markersize=1, color='0.3', alpha=0.4)

    ax.quiver(ds.s.sel(time=t_plot, space='x'), ds.s.sel(time=t_plot, space='y'),
              ds.v.sel(time=t_plot, space='x'), ds.v.sel(time=t_plot, space='y'),
              units='dots', angles='xy', width=2, scale_units='xy', color='0.3', alpha=0.4)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    if SAVE_FIG:
        fig.savefig('output/subfigures/frame_with_arrows.pdf', format='pdf', dpi=300)

#%% P'tit wrap


def wrapper_plot(paths, filename_timeseries):

    plt.close('all')
    set_matplotlib_config(presentation=True)

    ##
    #ds = dataloader_multiple(paths)
    #fig_general_plot(ds)
    #figure_density_centered_on_centroid(ds, 6, histo_n_bins=100)

    ##
    ds = dataloader(filename_timeseries)
    fig_timeserie_graph(ds)
    #fig_timeserie_snapshots(ds)
    # fig_example_frame(ds)

    ##
    plt.tight_layout()
    plt.show()


#%% Main
if __name__ == "__main__":
     

    global SAVE_FIG; SAVE_FIG = True

    paths = glob.glob('cleaned/*/**/2022*/*/', recursive=True) + \
        glob.glob('cleaned/*/**/2021-12-21*/*/', recursive=True)
    path_timeserie = 'cleaned/3_VarLight/2022-01-06/1/trajectory.nc'

    wrapper_plot(paths, path_timeserie)
