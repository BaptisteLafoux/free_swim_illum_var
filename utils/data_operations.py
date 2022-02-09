#!/usr/bin/env python
"""
This script provides useful funcs to all other scripts
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

#%% Math / dataprocessing boring functions


def group_dataset(ds, dim, n_bins=52):

    ds_group_avg = ds.groupby_bins(dim, bins=n_bins, precision=5).mean()
    ds_group_avg = ds_group_avg.rolling_exp(light_bins=3, center=True).mean()

    ds_group_std = ds.groupby_bins(dim, bins=n_bins, precision=5).std()
    ds_group_std = ds_group_std.rolling_exp(light_bins=3, center=True).mean()

    return ds_group_avg, ds_group_std


def interpolate_nans(initial_data):
    '''
    Remove the NaNs from a signal. A 1D linear interpolation is used. 

    Parameters
    ----------
    initial_data : np.array
        An array with NaNs inside that you want to remove.

    Returns
    -------
    initial_data : np.array
        The same array but without the NaNs : hurray ! 

    '''

    nans, x = np.isnan(initial_data) + ~np.isfinite(initial_data), lambda z: np.array(z).nonzero()[0]
    initial_data[nans] = np.interp(x(nans), x(~nans), initial_data[~nans])

    return initial_data


def compute_focal_values(ds):

    import progressbar
    import trajectorytools as tt

    '''
    NOT WORKING FOR N-N YET (too long...)
    Compute trajectories centered on each individuals and rotated in its direction

    Parameters
    ----------
    ds : (Dataset)
        .

    Returns
    -------
    ds_mod : (Dataset) 
        a modified dataset with new data add as variables.
    '''

    print('##### Computing focal values')

    print('Basis change and rotation\n')

    plt.pause(0.5)
    sr = np.empty((ds.n_frames, ds.n_fish, ds.n_fish, 2))
    vr = np.empty((ds.n_frames, ds.n_fish, ds.n_fish, 2))
    ar = np.empty((ds.n_frames, ds.n_fish, ds.n_fish, 2))
    er = np.empty((ds.n_frames, ds.n_fish, ds.n_fish, 2))

    thetar = np.empty((ds.n_frames, ds.n_fish, ds.n_fish))

    for focal in progressbar.progressbar(range(ds.n_fish)):

        thetar[:, focal, ...] = ds.theta - ds.theta[:, focal]
        sc_focal = ds.s - ds.s[:, focal, :]
        sr[:, focal, ...] = tt.fixed_to_comoving(sc_focal, ds.e[:, focal, :])
        er[:, focal, ...] = tt.fixed_to_comoving(ds.e, ds.e[:, focal, :])

    vr = np.gradient(sr, axis=0, edge_order=2) * ds.fps
    ar = np.gradient(vr, axis=0, edge_order=2) * ds.fps

    sr = xr.DataArray(data=sr, dims=['time', 'fish', 'neighbour', 'space'])
    vr = xr.DataArray(data=vr, dims=['time', 'fish', 'neighbour', 'space'])
    ar = xr.DataArray(data=ar, dims=['time', 'fish', 'neighbour', 'space'])
    er = xr.DataArray(data=er, dims=['time', 'fish', 'neighbour', 'space'])

    thetar = xr.DataArray(data=thetar, dims=['time', 'fish', 'neighbour'])

    ds_mod = ds.assign(sr=sr, vr=vr, ar=ar, er=er, thetar=thetar)

    print('#### Done computing focal values. They are added as new variables in the dataset\n')

    return ds_mod


def index_nn(ds, N=1, include_self=False):

    # A bit convoluted but it is for the sake of keeping the right dimension names in the xarray

    return np.argsort(ds.ii_dist, axis=-1)[..., 1-int(include_self):N+1]


def xarray_vector_norm(x, dim, ord=None):
    return xr.apply_ufunc(
        np.linalg.norm, x, input_core_dims=[[dim]], kwargs={"ord": ord, "axis": -1}
    )


def compute_corr_radius(ds, max_R=30, n_radii=100, n_timepts=500):
    '''
    See https://fr.wikipedia.org/wiki/Indice_de_Ripley

    Parameters
    ----------
    ds : Dataset
        DESCRIPTION.
    max_R : float, optional
        DESCRIPTION. The default is 30.
    n_radii : int, optional
        DESCRIPTION. The default is 100.
    n_timepts : int, optional
        DESCRIPTION. The default is 500.

    Returns
    -------
    corr_r : float array size n_timepts
        .
    time : float array size n_timepts
        .
    '''

    print('\n####### Computing correlation radius ######\n')
    print(f'Max radius : {max_R} BL, number of time points : {n_timepts}\n')

    import ripleyk

    radii = list(np.linspace(0, max_R, n_radii))

    times = ds.time[::ds.time.size//n_timepts]
    corr_r = np.empty_like(times)

    for i, t in enumerate(times):

        x, y = ds.s.sel(time=t).to_numpy().T
        k_func = np.array(ripleyk.calculate_ripley(radii, max_R, d1=x, d2=y,
                          boundary_correct=False, CSR_Normalise=False, sample_shape='circle'))

        H_func = np.sqrt(k_func / np.pi) - np.array(radii)

        corr_r[i] = radii[np.nanargmax(H_func)]

    return (corr_r, times)
