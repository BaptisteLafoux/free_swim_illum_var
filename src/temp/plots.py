#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:33:12 2022

@author: baptistelafoux
"""

#%% Full figures

def plot_all_variables_VS_light(ds): 
    
    fig, ax = plt.subplots(2, 1, figsize=(3, 4), sharex=True)
    fig.canvas.set_window_title('All data VS light')
    import palettable as pal
    col = pal.tableau.TableauMedium_10.mpl_colors
    #col = ['xkcd:coral', 'xkcd:raspberry', 'xkcd:tangerine']
    
    ds['vel'] = xarray_vector_norm(ds.v, dim='space')
    ds, ds_std = group_dataset(ds, dim='light', n_bins=36)
    
    ds = ds.rolling_exp(light_bins=2).mean()
    ds_std = ds_std.rolling_exp(light_bins=2).mean()

    # ax[0].plot(ds.light, ds.vel.mean(dim='fish'), color=col[0])
    # ax[0].fill_between(ds.light, ds.vel.mean(dim='fish') - ds_std.vel.mean(dim='fish'), ds.vel.mean(dim='fish') + ds_std.vel.mean(dim='fish'), alpha=0.2, color=col[0])
    # ax[0].set_ylabel('$|u|$ [BL/s]')
    
    ##
    

    ax1a = ax[0]; ax1b = ax[0].twinx()
    
    # IID
    avg_ii_dist = ds.ii_dist.mean(dim=['fish', 'neighbour'])
    
    ax1a.plot(ds.light, avg_ii_dist, '-', color=col[1])
    
    ax1a.set_ylabel('II-D [BL]', color=col[1])
    ax1a.tick_params(axis='y', labelcolor=col[1])
    
    # NND
    avg_nn_dist = ds.nn_dist.mean(dim='fish')
    
    ax1b.plot(ds.light, avg_nn_dist, '-', color=col[2])
    
    ax1b.set_ylabel('NN-D [BL]', color=col[2])
    ax1b.tick_params(axis='y', labelcolor=col[2])

    
    ##
    ii_dist_avg = ds.ii_dist.mean(dim=['neighbour', 'fish'])
    C = (np.min(ii_dist_avg) / ii_dist_avg)
    #C = 1
    
    ds['rot_param'] *= C; ds['pol_param_loc'] *= C
    
    ax[1].plot(ds.light, ds.pol_param_loc, 'o', label='$P_{\mathrm{loc}}$', color=col[3], markersize=4)
    ax[1].fill_between(ds.light, np.abs(ds.pol_param_loc) - ds_std.pol_param_loc, np.abs(ds.pol_param_loc) + ds_std.pol_param_loc, alpha=0.2, color=col[3])
    
    ax[1].plot(ds.light, np.abs(ds.rot_param), 'o', label='$M$', color=col[4], markersize=4)
    ax[1].fill_between(ds.light, np.abs(ds.rot_param) - ds_std.rot_param, np.abs(ds.rot_param) + ds_std.rot_param, alpha=0.2, color=col[4])
    
    # ax[2].plot(ds.light, np.abs(ds.rot_param_mod), '--o', label='$M$', color=col[5])
    # ax[2].fill_between(ds.light, np.abs(ds.rot_param_mod) - ds_std.rot_param_mod, np.abs(ds.rot_param_mod) + ds_std.rot_param_mod, alpha=0.2, color=col[5])
    
    ax[1].legend(loc=4)
    ax[1].set_xlabel('Illuminance [-]')
    ax[1].set_ylabel('[-]')
    
    fig.subplots_adjust(wspace=0)
    #plt.tight_layout()