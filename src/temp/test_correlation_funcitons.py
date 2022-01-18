#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:30:32 2022

@author: baptistelafoux
"""

plt.close('all')
from src.utilities import plot_frame_with_trajectories, add_illuminance_on_plot, dataloader, center_bins
import numpy as np
from scipy.signal import correlate

filename = 'cleaned/3_VarLight/2022-01-06/1/trajectory.nc'

DS = dataloader(filename)
DS.set_coords('light')

light_values = np.linspace(0, 1, 3)
fig, axs = plt.subplots(1, light_values.shape[0]-1, figsize=(20, 4)) 

for ld, lu, ax in zip(light_values[:-1], light_values[1:], axs): 
    
    ds = DS.sel(time=(ds.light==slice(ld, lu)))
    
    s_c = (ds.s - ds.center_of_mass).to_numpy().reshape((-1, 2))
    s = ds.s.to_numpy().reshape((-1,2))
    
    
    
    data, x, y = np.histogram2d(s[...,0], s[...,1], bins=40)
    x, y = center_bins(x, y)
    
    ax.contourf(x - x.mean(), y - y.mean(), data, cmap='Greys')
    ax.axis('scaled')
    
    
    # data, x, y = np.histogram2d(s_c[...,0], s_c[...,1], bins=30)
    # x, y = center_bins(x, y)
    
    
    # ax[1].contourf(x, y, data, cmap='Wistia')
    # ax[1].axis('scaled')
    # ax[1].axis(np.array([-ds.tank_size[0], ds.tank_size[0], -ds.tank_size[1], ds.tank_size[1]])/2)
    
    
    # ax[0].contourf(x, y, data, cmap='Reds', alpha=0.6)
    
plt.tight_layout()
plt.show() 