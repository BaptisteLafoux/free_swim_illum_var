import numpy as np
import cv2 
import time 
import glob 
import matplotlib.pyplot as plt 

from utils.loader import read_config

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

#%% Plot helpers

def set_matplotlib_config():
    
    import matplotlib as mpl 
    import warnings; warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
    import palettable as pal 
    
    config = read_config()
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use(config['viz'])
    
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=mpl.colors.ListedColormap(pal.tableau.Tableau_10.mpl_colors).colors)
    
def tight_all_opened_figures() :
    """
    DOESN'T WORK I DON'T KNOW WHY
    
    Parameters
    ----------
    
    figures: (list) figure numbers
    tight  : (bool) if True, applies the tight_layout option to all listed figures
    
    """
    figures = plt.get_fignums()
    
    for fig_id in figures:
        fig = plt.figure(fig_id)
        fig.tight_layout()
        
    plt.show()

def set_suptitle(title, fig, ds): 
    '''
    Self-exp
    
    '''
    
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

def add_illuminance_on_plot(ax, ds, scaling_factor=1, **kwargs):
    '''
    Add the light signal on an axe already plotted

    Parameters
    ----------
    ax : Axe
        An axe from matplotlib.
    ds : DataSet
        A dataset from fasttrack2xarray.py.

    Examples
    ::
        
        ds = dataloader(data_file_name)
        fig, ax = plt.subplots(2, 1)
        add_illuminance_on_plot(ax, ds)
        
    '''
    ax.plot(ds.time, ds.light * scaling_factor, '-', **kwargs)
    
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
    
def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

#%% Fancy movies

def get_moviename_from_dataset(ds, noBG=False):
    from pathlib import Path
    
    root = str(Path(ds.track_filename).parents[1])
    date = ds.date
    
    if noBG: movie_filename = f'{root}/{date}_noBG.mp4'
    else: movie_filename = f'{root}/{date}.mp4'
    
    return movie_filename
    
def plot_frame_with_trajectories(ds, ax, t, tr_len=12, noBG=True, **kwargs):
    '''
    

    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    tr_len : TYPE, optional
        DESCRIPTION. The default is 12.
    noBG : TYPE, optional
        DESCRIPTION. The default is True.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    frame : TYPE
        DESCRIPTION.

    '''
    
    movie_filename = get_moviename_from_dataset(ds, noBG=True)
    
    cap = cv2.VideoCapture(movie_filename)
    
    i = int(np.abs(ds.time - t).argmin())
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, i-1) 
    
    _, frame = cap.read()
    
    add_traj(ds, ax, i, tr_len, **kwargs)
    
    ax.imshow(frame, cmap='Greys_r')
    ax.axis('off')
    
    return frame

def generate_linecollection(ds, ax, i, tr_len, **kwargs):
    from matplotlib.collections import LineCollection
    
    widths = np.linspace(0, 2, tr_len) 
    
    LC = []
    
    for f in ds.fish:     
        points = np.array([ds.s[i-tr_len:i, f, 0], ds.s[i-tr_len:i, f, 1]]).T.reshape(-1, 1, 2) * ds.mean_BL_pxl
        
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, linewidths=widths, **kwargs)
        
        LC.append(lc)
    return LC
        
def add_traj(ds, ax, i, tr_len, **kwargs):
    '''
    

    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    tr_len : TYPE
        DESCRIPTION.
    noBG : TYPE, optional
        DESCRIPTION. The default is True.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    LC = generate_linecollection(ds, ax, i, tr_len, **kwargs)
    for lc in LC:
        ax.add_collection(lc)



