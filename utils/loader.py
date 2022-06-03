# -*- coding: utf-8 -*-

import xarray as xr
from utils.data_operations import add_modified_rot_param, compute_focal_values

def read_config():
    '''
    Reads the config.yaml file

    Returns
    -------
    config : dict

    '''
    import yaml

    config = {k: v for d in yaml.load(
        open('config.yaml'),
        Loader=yaml.SafeLoader) for k, v in d.items()}
    return config


def dataloader(data_file_name):

    ds = xr.open_dataset(data_file_name)

    return ds


def dataloader_multiple(paths, T_av=1, T_add=100):

    from time import perf_counter

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
        #ds = add_modified_rot_param(ds, L=12)

        ds = ds.coarsen(time=int(T_av*ds.fps), boundary='trim').mean()
        #ds = compute_focal_values(ds)

        loaded_ds.append(ds)
        attr.append(ds.attrs)

    DS = xr.concat(loaded_ds, dim='experiment', combine_attrs='drop', compat='no_conflicts')

    attr = {k: [dic[k] for dic in attr] for k in attr[0]}
    DS.attrs.update(attr)
    DS.attrs['fps'] = 1/T_av

    t2 = perf_counter()
    print(f'\nMerging all {len(paths)} datasets took {t2-t1:.2f} s')
    return DS
