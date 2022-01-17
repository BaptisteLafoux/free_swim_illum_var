#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 21:27:31 2022

@author: baptistelafoux
"""

plt.close('all')
from src.utilities import plot_frame_with_trajectories, add_illuminance_on_plot

import string; alph = string.ascii_uppercase

times = [140, 800, 1000, 2500]
times = np.array(times)

###
fig, ax0 = plt.subplots() 

ds.plot.scatter(x='time', y='rot_param', linestyle='-', marker=None)

add_illuminance_on_plot(ax0, ds, scaling_factor=1, color='C3')

###
fig, axs = plt.subplots(times.shape[0], 1, figsize=(4, 20)) 


for i, (ax, t) in enumerate(zip(axs, times)):
    plot_frame_with_trajectories(ds, ax, t=t)
    
    ax.text(x=50, y=ds.tank_size[1]*ds.mean_BL_pxl-50, s=alph[i], color='w', fontsize=16)
    
    ax0.plot(t, ds.rot_param[ds.time==t], 'ko', markersize=10, mfc="None")
    ax0.annotate(alph[i], (t, ds.rot_param[ds.time==t]), xytext=(10, 10), textcoords='offset pixels')


plt.tight_layout()
plt.show()
