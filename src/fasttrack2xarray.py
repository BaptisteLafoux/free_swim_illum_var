#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 16:48:57 2021

This script clean data from FastTrack analysis to crate an Xarray object that is saved in an image folder, in a .nc file 

@author: baptistelafoux
"""

#### WARNING : TO  DO - Slide light signal time basis to account for different starting time 

# %% Modules import
import numpy as np
import pandas as pd
import cv2
import os
import glob
import time as time_module
import xarray as xr
import yaml
import termplotlib as tpl
import matplotlib.pyplot as plt
import src.utilities as utils


from scipy.spatial.distance import pdist, squareform
from datetime import timedelta


np.seterr(all="ignore")

# %% Main function


def generate_dataset(data_folder):

    # useful file names
    fasttrack_filename = glob.glob(data_folder + "/Tracking_Result*" + "/tracking.txt")
    if fasttrack_filename: 
        fasttrack_filename = fasttrack_filename[0]
    
        bg_filename = glob.glob(data_folder + "/Tracking_Result*" + "/background.pgm")[0]

    #### Extraction of general data, metadata & BG
    tic = time_module.time()

    # load raw data
    data = pd.read_csv(fasttrack_filename, sep="\t")
    print("Successfully loaded data from the server")

    # loading the background image
    bg = cv2.imread(bg_filename)
    


    # compute the mean size of fish (they are decomposed into head + tail)
    global mean_BL_pxl
    mean_BL_pxl = np.mean(2 * data.headMajorAxisLength +
                          2 * data.tailMajorAxisLength)

    n_frames = np.max(data.imageNumber) + 1
    n_fish = np.max(data.id) + 1
    
    fish_by_frame = [data[data.imageNumber == i].shape[0] for i in range(n_frames)]
    tank_w, tank_h = bg.shape[1], bg.shape[0]
    
    #### Log infos
    print('\n#########################')
    print("Number of fish: ", n_fish)
    print("Number of frames: ", n_frames,
          '(' + str(timedelta(seconds=int(n_frames/fps))) + ' @ {:.0f} fps)'.format(fps))
    print("Avg. body length in pxl: {:.2f} pxl \n".format(mean_BL_pxl))
    print('#########################\n')

    toc = time_module.time()
    print("Loading file, metadata and BG \t {:.2f} s".format(toc - tic))

    #### Coordinates interpolation
    tic = time_module.time()

    s, v, a, e, theta, vel = generate_traj(data, n_frames, n_fish)
    toc = time_module.time()
    print(
        "Coordinates and coordinates interpolation \t {:.2f} s".format(toc - tic))

    #### Distances and rotation/polarization parameters
    tic = time_module.time()

    ii_dist = np.array([squareform(pdist(s[t, ...])) for t in range(n_frames)])
    if n_fish==1: 
        return None, False
    else: 
        nn_dist = np.sort(ii_dist, axis=1)[:, 1]

        center_of_mass = np.mean(s, axis=1)
        
        if circular_arena:
            center = find_arena_center(bg)  
            r = s - center[None, None, :] / mean_BL_pxl
        else:
            r = s - center_of_mass[:, None, :]
    
        rotation_parameter = rot_param(r, v, n_fish)
        
        polarization_parameter = pol_param(e, n_fish)

    toc = time_module.time()
    print(
        "Distances and rotation/polarization parameters \t {:.2f} s".format(
            toc - tic)
    )

    #### Create the dataset
    attrs = {
        "track_filename": fasttrack_filename,
        "bg_filename": bg_filename,
        "n_frames": n_frames,
        "n_fish": n_fish,
        "fps": fps,
        "mean_BL_pxl": mean_BL_pxl,
        "tank_size": pxl2BL(np.array([tank_w, tank_h]))
    }

    data_dict = dict(
        s=(["time", "fish", "space"], s),
        v=(["time", "fish", "space"], v),
        a=(["time", "fish", "space"], a), 
        vel=(["time", "fish"], vel),
        center_of_mass=(["time", "space"], center_of_mass),
        ii_dist=(["time", "fish", "neighbour"], ii_dist),
        nn_dist=(["time", "fish"], nn_dist),
        e=(["time", "fish", "space"], e),
        theta=(["time", "fish"], theta),
        rot_param=(["time"], rotation_parameter),
        pol_param=(["time"], polarization_parameter),
        fish_by_frame=(["time"], fish_by_frame)
    )

    coords = {
        "time": (["time"], np.arange(n_frames) / fps),
        "fish": (["fish"], np.arange(n_fish)),
        "neighbour": (["neighbour"], np.arange(n_fish)),
        "space": (["space"], ["x", "y"]),
    }
    
    #### metadata.yaml file 
    # only if the file exists (should be the case for all experiments after septembre 2021)
    
    metadata_filename = glob.glob(data_folder + "/metadata*")
    
    if metadata_filename: 
        
        print('\n#### There is a metadata.yaml file : processing it')
        metadata_filename = metadata_filename[0]
        time = np.arange(n_frames) / fps

        file = open(metadata_filename)
        metadata = yaml.load(file, Loader=yaml.FullLoader)
        file.close()
        print("Successfully loaded metadata", "\n")

        # we interpolate the light signal from metadata in case it is not an the same rate
        light = np.interp(
            np.linspace(0, len(metadata["light"]), n_frames),
            np.arange(len(metadata["light"])),
            metadata["light"],
        )

        print('Exact start time :', str(metadata['t_start']))

        fig = tpl.figure()
        fig.plot(time, light, width=60, height=10)
        fig.show()
        
        attrs_meta = {
            "T_exp": metadata["T_exp"],
            "T_settle": metadata["T_settle"],
            "T_period": metadata["T_per"],
            "date": str(metadata["t_start"].date()),
            "t_start": str(metadata["t_start"].time()),
        }
        
        attrs.update(attrs_meta)
        
        data_dict['light'] = (["time"], light)
        
        
    ds = xr.Dataset(data_vars=data_dict, coords=coords, attrs=attrs)
    
    print("Dataset generated without too big of a surprise")
    return ds, True


# %% Utilities function


def generate_traj(data, n_frames, n_fish):
    """
    A function that generates velocity and acceleration data from dirty (x,y) data. 
    It interpolates the data so that they are all on the same time basis, remove NaNs

    All values returned in BL !! 

    Parameters
    ----------
    data : DataSet 
        Generated from Fasttrack tracking.txt file.
    n_frames : int
    n_fish : int
    time : np.array of int
        Common time basis.

    Returns
    -------
    s : np.array
        size : (n_frames, n_fish, 2). (x, y) position for each frame for each fish.
    v : np.array
        (n_frames, n_fish, 2). (v_x, v_y) velocity for each frame for each fish.
    a : np.array
        (n_frames, n_fish, 2). (a_x, a_y) acceleration for each frame for each fish.
    e : np.array
        heading vector.
    theta : np.array
        angle with vertical (<- not sure for the with vertical part, it is an angle though).

    """
    
    time = np.arange(n_frames)
    
    s = np.empty((n_frames, n_fish, 2))
    e = np.empty((n_frames, n_fish, 2))
    theta = np.empty((n_frames, n_fish))

    for focal in range(n_fish):

        t = data[data.id == focal].imageNumber
        x = data[data.id == focal].xHead
        y = data[data.id == focal].yHead

        th = data[data.id == focal].tHead

        x_interp = np.interp(time, t, x)
        y_interp = np.interp(time, t, y)
        th_interp = np.interp(time, t, th)

        s[:, focal, :] = pxl2BL(np.c_[x_interp, y_interp])
        e[:, focal, :] = np.c_[np.cos(th_interp), np.sin(th_interp)]
        theta[:, focal] = th_interp

    v = np.gradient(s, axis=0, edge_order=2)
    a = np.gradient(v, axis=0, edge_order=2)
    
    vel = np.linalg.norm(v, axis=-1)
    
    s = utils.interpolate_nans(s)
    v = utils.interpolate_nans(v)
    a = utils.interpolate_nans(a)
    e = utils.interpolate_nans(e)
    vel = utils.interpolate_nans(vel)
    
    theta = utils.interpolate_nans(theta)

    return s, v, a, e, theta, vel


def rot_param(r, v, N):

    # we add an epilon to the norm of the velocity in case is it 0
    rotation_parameter = np.sum(np.cross(r, v) / (np.linalg.norm(r, axis=2) * (np.linalg.norm(v, axis=2) + 10**(-9)) ), axis=1) / N
    rotation_parameter = utils.interpolate_nans(rotation_parameter)

    return rotation_parameter
                                  

def pol_param(e, N):

    polarization_parameter = np.linalg.norm(np.sum(e, axis=1), axis=1) / N
    polarization_parameter = utils.interpolate_nans(polarization_parameter)

    return polarization_parameter


def pxl2BL(value):
    ''' dumb '''
    return value / mean_BL_pxl

def find_arena_center(bg):
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(bg, cv2.HOUGH_GRADIENT, 1, bg.shape[0] / 8,
                          param1=100, param2=30,
                          minRadius=100, maxRadius=1000)
    
    if circles.shape[0] > 1: print('\nWarning : found more than one circle for arena center detection\n')
    
    return circles[0,0,0:2]


def create_folder_cleandata(data_folder):
    
    relpath = os.path.dirname(os.path.relpath(data_folder, root + "1_raw_data"))

    # create a folder to store the cleaned data
    cleandata_foldername = root + "2_cured_datasets/" + relpath
    
    if not os.path.exists('cleaned/' + relpath):
        os.makedirs('cleaned/' + relpath)
    if not os.path.exists(cleandata_foldername):
        os.makedirs(cleandata_foldername)
    
    return cleandata_foldername, relpath

def save_xarray_file(dataset, save_cloud=True, save_loc=False):

    print("\n")
    print("Saving file... (can take a while)")
    tic = time_module.time()
    
    cleandata_foldername, relpath = create_folder_cleandata(data_folder)
    
    if save_cloud: 
        dataset.to_netcdf(cleandata_foldername + "/trajectory.nc", mode='w')
        
    if save_loc:
        dataset.to_netcdf('cleaned/' + relpath + '/trajectory.nc', mode='w')

    toc = time_module.time()
    print("File saved in \t {:.2f} s".format(toc - tic))
    print("Trajectory file saved in :", cleandata_foldername)


def save_control_plots(ds):
    '''Plot some stuffs to check if the traj. data makes sense.'''
    
    cleandata_foldername, _ = create_folder_cleandata(data_folder)

    plt.close('all')

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle(cleandata_foldername)
    
    #### [plot] hist. of nb. of detected fish 
    ax[0, 0].hist(ds.fish_by_frame, bins=ds.n_fish, color='k')
    ax[0, 0].set_xlabel('Nb detected fish')
    ax[0, 0].set_ylabel('Nb frames')

    #### [plot] nb. of detected fish over time 
    ax[0, 1].plot(ds.time / 60, ds.fish_by_frame, 'k.', markersize=0.2)
    ax[0, 1].set_xlabel('Time [min]')
    ax[0, 1].set_ylabel('Nb detected fish')

    #### [plot] all the trajectories
    ax[0, 2].plot(ds.s[..., 0], ds.s[..., 1], '.', markersize=0.2)
    ax[0, 2].axis('scaled')
    ax[0, 2].set_xlabel('[BL]')
    ax[0, 2].set_ylabel('[BL]')
    ax[0, 2].set_title('All traj')

    #### [plot] one random trajectory 
    random_index = np.random.randint(0, ds.n_fish)
    ax[1, 0].plot(ds.s[:, random_index, 0], ds.s[:, random_index, 1], 'k-')
    ax[1, 0].axis('scaled')
    ax[1, 0].set_xlabel('[BL]')
    ax[1, 0].set_ylabel('[BL]')
    ax[1, 0].set_title('On random traj. (fish n°' + str(random_index) + ')')

    plt.tight_layout()
    
    plt.pause(0.1); plt.plot()
    
    fig.savefig(cleandata_foldername + '/check_plots.png')
    
    print('\n')
    print('Control plots saved in ', cleandata_foldername)
    
    
def clean_data(data_folder):
    '''
    A wrapper

    Parameters
    ----------
    data_folder : str

    Returns
    -------
    The clean dataset generated and its location 

    ''' 
    print("\n")
    print(" Processing data ".center(80, "*"), "\n")

    print("Data folder :", data_folder, "\n")
    
    ds, res = generate_dataset(data_folder)
    
    if res:
        save_xarray_file(ds, save_cloud=True, save_loc=False) 
        save_control_plots(ds)
    
    return ds, create_folder_cleandata(data_folder)

# %% Run code
if __name__ == "__main__":

    global fps, root, circular_arena
    fps = 5
    root = '/Volumes/baptiste/data_labox/illuminance_variation/' #'/Volumes/baptiste/2019_PMMH_stage_AGimenez_collective/Collective/1_data_alicia/'#
    circular_arena = False
    
    
    root_data = f'{root}1_raw_data/3_VarLight' #root#
    
    folder_list = glob.glob(f"{root_data}*/**/Tracking*", recursive = True)
    
    for data_folder in folder_list:
        
        _, _ = clean_data(os.path.dirname(data_folder))
        
        
        
        
        