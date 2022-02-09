#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 10:11:16 2022

@author: baptistelafoux
"""

import glob
import shutil
import os
import cv2
import matplotlib.pyplot as plt
import progressbar

from utils.loader import dataloader
from utils.graphic import generate_linecollection, set_matplotlib_config, get_moviename_from_dataset

from pathlib import Path

from matplotlib.animation import FFMpegWriter

paths = glob.glob('cleaned/*/**/2022-01-06/*/trajectory.nc', recursive=True)
set_matplotlib_config()


for path in paths:
    if os.path.isdir('cache') : shutil.rmtree('cache')
    
    ds = dataloader(path)
    root = str(Path(ds.track_filename).parents[1])

    original_movie_filename = get_moviename_from_dataset(ds, noBG=False)
    save_movie_filename = f'{os.path.dirname(path)}/{ds.date}_traj.mp4'
    
    writer = FFMpegWriter(fps=ds.fps)

    os.makedirs('cache')
    shutil.copy(original_movie_filename, 'cache/')

    cap = cv2.VideoCapture(glob.glob('cache/*.mp4')[0])

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    dpi = 100
    dim = (h / dpi, w / dpi)
    
    fig, ax = plt.subplots(figsize=dim)

    _, frame = cap.read()
    im = ax.imshow(frame, aspect='equal')
    
    ax.axis('scaled')
    ax.axis('off')
    
    ax.set_position([0, 0, 1, 1])
    
    #fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    with writer.saving(fig, save_movie_filename, dpi):

        for i in progressbar.progressbar(range(1, n_frames)):

            cap.set(cv2.CAP_PROP_POS_FRAMES, i-1)
            _, frame = cap.read()

            im.set_data(frame)
            
            LC = generate_linecollection(ds, ax, i, tr_len=12, color='C3')
            
            for lc in LC: ax.add_collection(lc)
            
            writer.grab_frame()
            
            for lc in LC: lc.remove()
            

