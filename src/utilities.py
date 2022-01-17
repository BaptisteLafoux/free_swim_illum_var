#!/usr/bin/env python
"""
This script provides useful funcs to all other scripts
"""
import yaml
import cv2
import glob
import numpy as np
import time
import progressbar
import trajectorytools as tt 
import xarray as xr
import matplotlib.pyplot as plt 

import matplotlib as mpl
import palettable as pal

from time import perf_counter

#%% Background finding

def find_bg(filePath, nb_eval=5, mode='max'):
    '''
    Finds the background for a series of images by picking random samples.
    Only works if the things (fish) you want to observe move everywhere : there must be no overlapping in all selected iamges.

    Parameters
    ----------
    filePath : str
        DESCRIPTION.
    nb_eval : int, optional
        DESCRIPTION. The default is 5.
    mode : str, optional
        Can be 'median', 'max, 'min', or 'average'. The default is 'max'.

    Returns
    -------
    im_background : np.array of uint8
        Image with same dimensions as initial movie (H, W, nb of channels).

    '''

    print('###############################')
    print('Starting background evaluation')
    print('Mode : ' + mode)
    print('Samples : {}'.format(nb_eval))
    print('###############################')

    t0 = time.time()

    video_loader = cv2.VideoCapture(filePath)

    w = int(video_loader.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_loader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    numFrames = int(video_loader.get(cv2.CAP_PROP_FRAME_COUNT))

    # background substraction
    i_eval = np.linspace(0, numFrames-1, nb_eval).astype(np.uint32)

    buff = np.empty((h, w, 3, nb_eval))

    for i in range(nb_eval):
        video_loader.set(cv2.CAP_PROP_POS_FRAMES, i_eval[i])
        _, buff[:, :, :, i] = video_loader.read()
        print(i + 1, '/', nb_eval)

    if mode == 'median':
        im_background = np.nanmedian(buff, axis=3).astype(np.uint8)
    if mode == 'max':
        im_background = np.nanmax(buff, axis=3).astype(np.uint8)
    if mode == 'min':
        im_background = np.nanmin(buff, axis=3).astype(np.uint8)
    if mode == 'average':
        im_background = np.nanmean(buff, axis=3).astype(np.uint8)

    print('###############################')
    print('Background evaluation done in {:.2f} s'.format(time.time() - t0))
    print('###############################\n')

    return im_background


def find_bg_imageseries(folder_path, nb_eval=5, mode='max', extension='tiff'):
    '''
    Finds the background for a series of images by picking random samples.
    Only works if the things (fish) you want to observe move everywhere : there must be no overlapping in all selected iamges.

    Parameters
    ----------
    filePath : str
        DESCRIPTION.
    nb_eval : int, optional
        DESCRIPTION. The default is 5.
    mode : str, optional
        Can be 'median', 'max, 'min', or 'average'. The default is 'max'.

    Returns
    -------
    im_background : np.array of uint8
        Image with same dimensions as initial movie (H, W, nb of channels).

    '''

    print('###############################')
    print('Starting background evaluation')
    print('Mode : ' + mode)
    print('Samples : {}'.format(nb_eval))
    print('###############################')

    t0 = time.time()

    files = np.sort(glob.glob(folder_path + '/*.' + extension))

    im = cv2.imread(files[0])
    numFrames = len(files)
    buff = np.repeat(np.empty_like(im)[..., np.newaxis], nb_eval, axis=-1)

    # background substraction
    i_eval = np.linspace(0, numFrames-1, nb_eval).astype(np.uint32)

    for i in range(nb_eval):

        im = cv2.imread(files[i_eval[i]])
        buff[..., i] = im
        print(i + 1, '/', nb_eval)

    if mode == 'median':
        im_background = np.nanmedian(buff, axis=3).astype(np.uint8)
    if mode == 'max':
        im_background = np.nanmax(buff, axis=3).astype(np.uint8)
    if mode == 'min':
        im_background = np.nanmin(buff, axis=3).astype(np.uint8)
    if mode == 'average':
        im_background = np.nanmean(buff, axis=3).astype(np.uint8)

    print('###############################')
    print('Background evaluation done in {:.2f} s'.format(time.time() - t0))
    print('###############################\n')

    return im_background

#%% Math / dataprocessing boring functions

def group_dataset(ds, dim, n_bins=52):
    
    ds_group_avg = ds.groupby_bins(dim, bins=n_bins, precision=5).mean()
    ds_group_avg = ds_group_avg.rolling_exp(light_bins=3).mean()
    
    ds_group_std = ds.groupby_bins(dim, bins=n_bins, precision=5).std()
    ds_group_std = ds_group_std.rolling_exp(light_bins=3).mean()
    
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

def add_pol_param_local_to_ds(ds, N):
    '''
    WARNING !!! deprecated for dataset with multiple experiments (do not use)

    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.

    Returns
    -------
    ds : TYPE
        DESCRIPTION.

    '''
    
    i_nn = index_nn(ds, N, include_self=True).to_numpy()
    i_nn = np.repeat(i_nn[..., None], ds.space.size, axis=-1)
    
    e_loc = np.take_along_axis(ds.e.to_numpy()[..., None, :], i_nn, axis=-3)   
    
    pol_param_loc = np.linalg.norm(e_loc.mean(axis=-2), axis=-1).mean(axis=-1)
    
    if len(pol_param_loc.shape) > 1:
        dict_pol_param_loc = xr.DataArray(data=pol_param_loc, dims=['experiment', 'time'])
    else: 
        dict_pol_param_loc = xr.DataArray(data=pol_param_loc, dims=['time'])
        
    ds = ds.assign(pol_param_loc=dict_pol_param_loc)
     
    return ds

def add_rot_param_modif_to_ds(ds):
    
    r = ds.s - ds.center_of_mass.to_numpy()[..., None, :]
    w, h = ds.tank_size
    R0 = ds.ii_dist.min(dim=['time', 'neighbour', 'fish']).to_numpy()[..., None]
    
    rot_param_mod = rot_param_modif(r, ds.v, ds.fish.size, R0, ds) 
    
    if len(rot_param_mod.shape) > 1:
        dict_rot_param_mod = xr.DataArray(data=rot_param_mod, dims=['experiment', 'time'])
    else: 
        dict_rot_param_mod = xr.DataArray(data=rot_param_mod, dims=['time'])
        
    ds = ds.assign(rot_param_mod=dict_rot_param_mod)
     
    return ds

def rot_param_modif(r, v, N, R0, ds):

    norm_v = np.linalg.norm(v, axis=-1) + 10**(-9)
    norm_r = np.linalg.norm(r, axis=-1)

    #max_r_norm = np.repeat(np.nanmax(norm_r, axis=-1)[..., None], N, axis=-1)


    rotation_parameter_original = np.cross(r, v) / (norm_v * norm_r)
    
    
    
    rotation_parameter_modif = rotation_parameter_original * R0 / (R0 + ds.ii_dist.mean(dim='neighbour'))
    
    return rotation_parameter_modif.mean(axis=-1)
    
def compute_mean_vel_norm(ds):
    vel = xarray_vector_norm(ds.v, dim='space').mean(dim='fish')
    
    return vel 

#%% Dataloaders  
def read_config():
    '''
    Reads the config.yaml file

    Returns
    -------
    config : dict

    '''
    config = {k: v for d in yaml.load(
        open('config.yaml'),
        Loader=yaml.SafeLoader) for k, v in d.items()}
    return config

def dataloader(data_file_name):
    
    ds = xr.open_dataset(data_file_name)
    
    return ds

def dataloader_multiple(paths, T_av=1, T_add=0):
    xr.set_options(keep_attrs=True)
    
    print('################ Creating a large dataset to gather multiple experiments ################\n')
    
    print(f'///// WARNING : here we average all variables over {T_av} s \\\\\\\\ \n')
    t1 = perf_counter()
    
    loaded_ds = []
    attr = []
    
    for i, path in enumerate(paths): 
        
        print(f'Merging file : cleaned/3_VarLight/{path}/trajectory.nc - {i+1}/{len(paths)}')
        
        ds = dataloader(f'{path}/trajectory.nc')
        ds = ds.sel(time=slice(ds.T_settle - T_add, ds.T_settle + ds.T_exp))
        ds = ds.assign_coords(time=ds.time - (ds.T_settle - T_add))
        
        #ds = add_pol_param_local_to_ds(ds, N=15)
        #ds = add_rot_param_modif_to_ds(ds)
        
        ds = ds.coarsen(time=int(T_av*ds.fps), boundary='trim').mean()
        
        loaded_ds.append(ds)
        attr.append(ds.attrs)
    
    DS = xr.concat(loaded_ds, dim='experiment', combine_attrs='drop', compat='no_conflicts')
    
    attr = {k: [dic[k] for dic in attr] for k in attr[0]}
    DS.attrs.update(attr)
    DS.attrs['fps'] = 1/T_av
    
    t2 = perf_counter()
    print(f'\nMerging all {len(paths)} datasets took {t2-t1:.2f} s')
    return DS
    
#%% Plot helpers

def set_matplotlib_config():
    

    
    config = read_config()
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use(config['viz'])
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=mpl.colors.ListedColormap(pal.tableau.BlueRed_6.mpl_colors).colors)
    
def tight_all_opened_figures() :
    """
    DOESN'T WORK I DON'T KNOW WHY
    Maximize all the open figure windows or the ones indicated in figures
    
    Parameters
    ----------
    
    figures: (list) figure numbers
    tight  : (bool) if True, applies the tight_layout option
    
    """
    figures = plt.get_fignums()
    
    for fig_id in figures:
        fig = plt.figure(fig_id)
        fig.tight_layout()
        
    plt.show()

def set_suptitle(title, fig, ds): 
    
    fig.canvas.set_window_title(title)
    
    if 'experiment' in ds.rot_param.dims: 
        fig.suptitle(f'{title} [{len(ds.experiment)} exps.]') 
    else:
        fig.suptitle(f'{ds.date} ({ds.n_fish} fish)\n {title}')
        
    
def center_bins(x_bins, y_bins):
    '''
    Convert bin values that are originally on the side to the center of the bin (you don't understand ? see examples)
    It's useful for plt.contour
    
    Parameters
    ----------
    x_bins : (ndarray)
        size N.
    y_bins : (ndarray)
        size N.

    Returns
    -------
    x_center : (ndarray)
        Achtung : size N-1.
    y_center : (ndarray)
        Achtung : size N-1.

    Example
    -------
    ::
        
        p, x_bins, y_bins = np.histogram2d(y, x)
        x_center, y_center = center_bins(x_bins, y_bins)
        
        plt.contour(x_center, y_center, p)


    '''
    x_center = x_bins[:-1] + np.diff(x_bins) / 2
    y_center = y_bins[:-1] + np.diff(y_bins) / 2
    
    return (x_center, y_center)

def add_illuminance_on_plot(ax, ds, scaling_factor=1, color='C3'):
    '''
    Add the light signal on an axe already plotted

    Parameters
    ----------
    ax : Axe
        An axe from matplotlib.
    ds : DataSet
        A dataset from fasttrack2xarray.py.

    Examples
    --------
    ::
        
        ds = dataloader(data_file_name)
        fig, ax = plt.subplots(2, 1)
        add_illuminance_on_plot(ax, ds)
        
    '''
    ax.plot(ds.time, ds.light * scaling_factor,  '-', color=color, label='Illuminance')
    
def save_multi_image(filename, rasterized=True, sep=True):
    
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt 
    import os
    
    import warnings
    warnings.filterwarnings("ignore")
    
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    
    for fig in figs:
        for ax in fig.get_axes(): ax.set_rasterized(rasterized) 
        if sep : 
            title = fig.canvas.get_window_title()
            fig.savefig(f'output/{title}.pdf', format='pdf', transparent=True)
        else:
            unique_file = os.path.join(os.path.dirname(filename), 'plots.pdf')
            pp = PdfPages(unique_file)
            fig.savefig(pp, format='pdf', transparent=True)
            
    if not sep : pp.close()
            
     
    
def set_colorbar_right(ax, plot, cb_width_percent=5, cb_dist_percent=5, position='right'):
    '''
    Position the colorbar so that it has the same size as the plot and constant distance to the plot when using plt.tight_layout() 

    Parameters
    ----------
    ax : AxesSubplot
        DESCRIPTION.
    plot : output of plt.plot() 
        DESCRIPTION.
    cb_width_percent : (int) 
        width of the colorbar in % of the plot size. The default is 5.
    cb_dist_percent : TYPE, optional
        DESCRIPTION. The default is 5.
    position : TYPE, optional
        DESCRIPTION. The default is 'right'.

    Example
    -------
    ::
        
        fig, ax = plt.subplots()
        plot = ax.pcolormesh(x, y, C)
        
        set_colorbar_right(ax, plot)
        plt.tight_layout()

    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, str(int(cb_width_percent)) + "%", pad=str(int(cb_dist_percent)) + "%")
    ax.colorbar(plot, cax=cax)    
    
#%% Fancy movies

def plot_frame_with_trajectories(ds, ax, t, tr_len=12):
    
    from pathlib import Path 
    from matplotlib.collections import LineCollection
    
    root = str(Path(ds.track_filename).parents[1])
    date = ds.date
    movie_filename = f'{root}/{date}.mp4'
    
    cap = cv2.VideoCapture(movie_filename)
    
    i = int(np.abs(ds.time - t).argmin())
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, i-1) 
    
    _, frame = cap.read()
    
    ax.imshow(frame, cmap='Greys_r')
    #ax.plot(ds.s[i, :, 0] * ds.mean_BL_pxl, ds.s[i, :, 1] * ds.mean_BL_pxl, 'ko')
    
    widths = np.linspace(0, 2, tr_len) 
    
    for f in ds.fish: 
        
        points = np.array([ds.s[i-tr_len:i, f, 0], ds.s[i-tr_len:i, f, 1]]).T.reshape(-1, 1, 2) * ds.mean_BL_pxl
        
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, linewidths=widths, color='C4')
        ax.add_collection(lc)
        
    ax.axis('off')


def movie_with_traj(ds, movie_full_path):
    '''
    To be continued

    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.
    movie_full_path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    from matplotlib.animation import FFMpegWriter
    import os 
    
    filename = os.path.basname(movie_full_path)
    writer = FFMpegWriter(ds.fps) 