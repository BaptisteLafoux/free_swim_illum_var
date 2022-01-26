#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:08:55 2022

@author: baptistelafoux
"""

from utils.loader import dataloader_multiple
from utils.graphic import save_multi_image

from src.analysis import analysis_plot
import glob 

def run_multi(paths):
    
    ds = dataloader_multiple(paths)
    
    ds_avg = ds.mean(dim='experiment', keep_attrs=True)
    analysis_plot(ds_avg, multi_analysis=True)

    save_multi_image(paths, rasterized=False, sep=True)
    
    return ds

#%% Main
if __name__ == "__main__":
    
    paths = glob.glob('cleaned/*/**/2022*/*/', recursive=True) + glob.glob('cleaned/*/**/2021-12-21*/*/', recursive=True) 
    
    ds = run_multi(paths)
    