#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 10:46:24 2022

@author: baptistelafoux
"""

import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

from scipy.ndimage import zoom, gaussian_filter
from utils.graphic import get_moviename_from_dataset
from utils.loader import dataloader

plt.close('all')
path_timeserie = 'cleaned/3_VarLight/2022-01-06/1/trajectory.nc'

ds = dataloader(path_timeserie)
movie_path = get_moviename_from_dataset(ds, noBG=False)

cap = cv2.VideoCapture(movie_path)

t = 41 * 60 * 5
t = int(t)

N = 30
_, frame_ini = cap.read() 

stack = frame_ini[None, ...]
np.zeros_like(frame_ini)[None, ...]

for i, t in enumerate(range(t-N, t, 1)): 
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, t-1)
    
    _, frame = cap.read() 
    
    stack = np.append(stack, frame[None,...], axis=0) 

stack = stack[1:]

fig, ax = plt.subplots(figsize=(15, 9)) 
final = stack.min(axis=0).astype(np.uint8)
cv2.normalize(final, final, 0, 255, cv2.NORM_MINMAX)

final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
  

final = gaussian_filter(zoom(final, 4) , sigma=0.8)

plt.imsave('output/pcfocus.png', 255 - final, cmap='Greys', dpi=300)

ax.axis('off')