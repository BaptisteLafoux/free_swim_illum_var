#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 16:23:24 2022

@author: baptistelafoux
"""

from utils.loader import dataloader
from src.analysis import add_var_vs_time
from utils.graphic import set_matplotlib_config 
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.animation import FuncAnimation, FFMpegWriter

#%%

def ini_pol_rot_timeserie(ax):
    
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('[-]')
    ax.set_ylim([0, 1])
    
    plot_obj = {}
    
    ax.plot(ds.time/60, ds.pol_param, lw=0.3, color='none')
    ax.plot(ds.time/60, ds.rot_param, lw=0.3, color='none')
    
    plot_obj['p_pol'], = ax.plot([], lw=1.5, color='C4', mfc='k', label='$\mathcal{P}$')
    plot_obj['p_rot'], = ax.plot([], lw=1.5, color='C3', mfc='k', label='$\mathcal{M}$')
    
    plot_obj['pt_pol'], = ax.plot([], '.', mew=1.5, mfc='C4', color='C4')
    plot_obj['pt_rot'], = ax.plot([], '.', mew=1.5, mfc='C3', color='C3')
    
    plot_obj['l'] = ax.axvline(0, ls='-', color='w', lw=0.75, zorder=0)
    
    plot_obj['leg'] = ax.legend(frameon=True, loc='upper left'); plot_obj['leg'].get_frame().set_linewidth(0.0)
    plot_obj['leg'].get_frame().set_facecolor('w')
    plot_obj['leg'].get_frame().set_alpha(0.3)
    
    return plot_obj

def ini_light_timeserie(ax):
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('[-]')
    
    plot_obj = {}
    
    ax.plot(ds.time / 60, ds.light, lw=0.5, color='w')
    
    plot_obj['p_light'],  = ax.plot([], lw=1.5, color='w', label=r'Illuminance $\bar{E}$')
    
    plot_obj['pt_light'], = ax.plot([], '.', mew=1.5, mfc='w', color='w')
    
    #plot_obj['l'] = ax.axvline(0, ls='-', color='0.4', zorder=0)
    
    plot_obj['leg'] = ax.legend(frameon=True, loc='upper left'); plot_obj['leg'].get_frame().set_linewidth(0.0)
    plot_obj['leg'].get_frame().set_facecolor('w')
    plot_obj['leg'].get_frame().set_alpha(0.3)
    return plot_obj

def remove_time_axis(ax):
    #ax.set_frame_on(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    return ax 

#%% 
set_matplotlib_config()

plt.style.use('dark_background')
plt.close('all')

import matplotlib.pyplot as mpl
mpl.rcParams['font.family'] = 'Avenir'


downsampling_period = 1


ds = dataloader('cleaned/3_VarLight/2022-01-06/1/trajectory.nc')
ds = np.abs(ds.coarsen(time=int(downsampling_period*ds.fps), boundary='trim').mean())

#ds = ds.rolling(time=120, center=True).mean()

fig, ax = plt.subplots(figsize=(4, 3)) 

fig.set_facecolor([35/255, 83/255, 137/255])
ax.set_facecolor([35/255, 83/255, 137/255])

#plot_obj = ini_pol_rot_timeserie(ax)
plot_obj = ini_light_timeserie(ax)
    


def animation_pol_rot_timeserie(i):
    
    dsi = ds.isel(time=slice(0, i+1))
    
    plot_obj['p_pol'].set_data(dsi.time / 60, dsi.pol_param)
    plot_obj['p_rot'].set_data(dsi.time / 60, dsi.rot_param)
    
    plot_obj['pt_pol'].set_data(dsi.time[-1] / 60, dsi.pol_param[-1])
    plot_obj['pt_rot'].set_data(dsi.time[-1] / 60, dsi.rot_param[-1])
    
    plot_obj['l'].set_xdata([dsi.time[-1] / 60, dsi.time[-1] / 60])
    
    if i%100 == 0: print(f'{i / ds.time.size * 100:5.2f} %')
    return plot_obj['leg'], plot_obj['pt_pol'], plot_obj['pt_rot'], plot_obj['p_pol'], plot_obj['p_rot'], plot_obj['l'], 

def animation_light_timeserie(i):
    dsi = ds.isel(time=slice(0, i+1))
    
    plot_obj['p_light'].set_data(dsi.time / 60, dsi.light)
    
    plot_obj['pt_light'].set_data(dsi.time[-1] / 60, dsi.light[-1])
    
    #plot_obj['l'].set_xdata([dsi.time[-1] / 60, dsi.time[-1] / 60])
    
    if i%100 == 0: print(f'{i / ds.time.size * 100:5.2f} %')
    return plot_obj['leg'], plot_obj['pt_light'], plot_obj['p_light'], #plot_obj['l'], 

    
    
    
anim = FuncAnimation(fig, animation_light_timeserie, frames=ds.time.size, blit=False, interval=10)  
anim.save('output/animated_light_vs_time_with_time_axis_blue_bg.mp4', dpi=120, writer=FFMpegWriter(fps=1/downsampling_period), savefig_kwargs=dict(facecolor=fig.get_facecolor()))

plt.show()
    
    
    
    