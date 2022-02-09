#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 10:46:24 2022

@author: baptistelafoux
"""

import cv2 
import numpy as np 

from utils.graphic import get_moviename_from_dataset
from utils.loader import dataloader

path_timeserie = 'cleaned/3_VarLight/2022-01-06/1/trajectory.nc'

ds = dataloader(path_timeserie)
movie_path = get_moviename_from_dataset(ds, noBG=False)

cap = cv2.VideoCapture(movie_path)

t = 3400

N = 10 
frame, _ = cap.read() 

stack = np.repeat(np.empty_like(frame)[None, ...], axis=0) 

for i, t in enumerate(range(t-10, t)): 
    cap.set(cv2.CAP_PROP_FRAME_COUNT)
    
    frame, _ = cap.read() 
    
    stack[i] = frame


