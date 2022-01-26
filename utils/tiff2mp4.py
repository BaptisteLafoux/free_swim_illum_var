# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 17:23:14 2021

@author: baptiste
"""


from glob import glob
import cv2
from progressbar import progressbar
import numpy as np
import natsort
import os
import findBackground

#%%
    
path = "/Volumes/baptiste/data_labox/films_influence_intensite_lumineuse/2021-12-21_tetra_10minrest_2cycles30min_monteedescente/1/"
directories = os.path.join(path, "data/")

print('#################')

    
files = natsort.natsorted(glob(directories + '/*.tiff'))
fps = 5        
bg = findBackground.findBackground_imageseries(path + 'data', 10, 'max')
frame = cv2.imread(files[0])
        
video_save_filename = path + "2021-20-12_frame_substracted"
writer = cv2.VideoWriter(video_save_filename + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
writer_noBG = cv2.VideoWriter(video_save_filename + '_noBG.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))


#%%

try:
    for file in progressbar(files): 
        frame = cv2.imread(file)
        writer.write(frame)
                
        frame_diff = cv2.bitwise_not(cv2.absdiff(frame, bg))
        cv2.normalize(frame_diff, frame_diff, 0, 255, cv2.NORM_MINMAX)
        

        writer_noBG.write(frame_diff.astype(np.uint8))
            
finally:
    writer.release()
    writer_noBG.release()