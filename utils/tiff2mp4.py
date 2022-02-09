# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 17:23:14 2021

@author: baptiste
"""
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import natsort
import os
import progressbar

from utils.graphic import find_bg_imageseries, run_once
from pathlib import Path

#%%


def init_writers(path):

    print(f'\n#### Initializing {path}\n')

    frames = natsort.natsorted(glob.glob(f'{path}/*.tiff'))

    bg = find_bg_imageseries(path, 10, 'max')

    vid_basename = f'{os.path.dirname(path)}/{os.path.basename(str(Path(path).parents[1]))}'

    frame_ini = cv2.imread(frames[0])
    w, h = frame_ini.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(f'{vid_basename}.mp4', fourcc, FPS, (w, h))
    writer_noBG = cv2.VideoWriter(f'{vid_basename}_noBG.mp4', fourcc, FPS, (w, h))

    return (frames, [writer, writer_noBG], bg)


@run_once
def show_frame(frame):

    fig, ax = plt.subplots()
    ax.imshow(frame, cmap='Greys_r')
    plt.show()
    plt.pause(0.1)


def generate_movie(frames, writers, bg):

    print('\n')
    plt.pause(1)

    writer, writer_noBG = writers

    try:
        for file in progressbar.progressbar(frames):

            frame = cv2.imread(file)
            ## Frame with BG
            writer.write(frame)
            ## Remove BG
            frame_diff = cv2.bitwise_not(cv2.absdiff(frame, bg))
            cv2.normalize(frame_diff, frame_diff, 0, 255, cv2.NORM_MINMAX)

            writer_noBG.write(frame_diff.astype(np.uint8))

            show_frame(frame_diff)

    finally:
        writer.release()
        writer_noBG.release()


if __name__ == "__main__":

    global FPS
    FPS = 5

    paths = glob.glob('/Volumes/baptiste/data_labox/illuminance_variation/1_raw_data/3_VarLight/2022-01-04/**/data')

    print(f'\nProcessing {len(paths)} movies')

    for path in paths:

        frames, writers, bg = init_writers(path)
        generate_movie(frames, writers, bg)
