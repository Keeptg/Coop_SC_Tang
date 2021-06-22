# This is a sample Python script.

import os
import mat73
from scipy.io import loadmat
import numpy as np
import pandas as pd
import xarray as xr
import salem

from path_config import *


def write_mat2nc(mat_tuple, varname, years, units=None, y0=None):
    """
    Wite the mat dataset to netcdf file.

    Params
    ------
    mat_tuple : dict-like: from opened *.mat file.
    varname : str: the variable name you want to operate.
        can be one of ['PrecipMean_ctl', 'PrecipMean_sce', 'T2mMean_ctl', 'T2mMean_sce']
    years : int: How many years the data need be repeated.
    units : None type or str: defalt is 'mm' for prec or 'km m-2 s-1'

    return
    ------
    mons_data : xarray.Dataset
    """

    day_interv = np.arange(0, 360, 30)
    day_data = mat_tuple[varname]
    if 'Precip' in varname:
        data_vars = 'pr'
    elif 'T2m' in varname:
        data_vars = 'tas'
    else:
        raise NameError("Can't understandable data type!")
    if data_vars == 'pr':
        if units == None:
            units = 'kg m-2 s-1'
            convert_fact = 30 * 24 * 60 * 60
        elif units == 'mm':
            convert_fact = 1
        else:
            raise NameError(f'No units named as {units}')
        month_data = [day_data[:, :, i:i + 30].sum(axis=2)/convert_fact for i in day_interv]
        long_name = 'Precipitation'
    elif data_vars == 'tas':
        if units == None:
            units = 'K'
            convert_fact = 0
        elif units == 'C':
            convert_fact = 273.15
        else:
            raise NameError(f'No units named as {units}')
        month_data = [day_data [:, :, i:i + 30].mean(axis=2) + convert_fact for i in day_interv]
        long_name = 'Near-Surface Air Temperature (2m)'
    else:
        raise NameError(f"No variable named as {varname}")
    num = 0
    while num < years:
        try:
            ys_data = np.append(ys_data, month_data, axis=0)
        except NameError:
            ys_data = month_data
        num += 1
    if y0 is None:
        y0 = '1/15/1950'
    time = pd.date_range(y0, periods=years * 12, freq='M')
    lon = mat_tuple['lon']
    lat = mat_tuple['lat']
    print(units)
    mons_data = xr.Dataset({data_vars: (["time", "lon", "lat"], ys_data)},
                          coords=dict(time=time, lon=lon, lat=lat),
                          attrs=dict(units=units, long_name=long_name))

    return mons_data


def write_climate2nc(years, file_name, **kwargs):

    for mtype in ['1', '2']:
        for etype in ['sce', 'ctl']:
            climate_path = os.path.join(data_dir, file_name)
            data = mat73.loadmat(climate_path)
            prec_ds = write_mat2nc(data, f'PrecipMean_{etype}', years, **kwargs)
            temp_ds = write_mat2nc(data, f'T2mMean_{etype}', years, **kwargs)
            prec_ds.to_netcdf(os.path.join(data_dir, f'prec{years}_{etype}_ds{mtype}.nc'))
            temp_ds.to_netcdf(os.path.join(data_dir, f'temp{years}_{etype}_ds{mtype}.nc'))


def write_climate_diff2nc(file_name, var_name, prcp_units=None, temp_units=None, suffix=''):
    """Write *.mat climate file to *.nc file
    Parameters
    ------
    file_name : str: the name of target file, should be ended with '.mat'

    """

    if suffix:
        suffix = '_' + suffix
    climate_path = os.path.join(data_dir, file_name)
    data = mat73.loadmat(climate_path)
    var_ds = write_mat2nc(data, var_name, 1)
    var_ds.to_netcdf(os.path.join(data_dir, f'{var_name}{suffix}.nc'))


def write_climate99_nc(var_name, outpath=None,
                       transform_method=None, long_name=None, group='ctl', savefile=True):
    """
    """

    fpath = os.path.join(data_dir, 'Climate2_99years')
    latlon_dict = mat73.loadmat(os.path.join(fpath, 'Climate2_99years_latlon.mat'))
    lat = latlon_dict['lat']
    lon = latlon_dict['lon']
    fname = f'Climate2_99years_{var_name}_{group}.mat'
    path = os.path.join(fpath, fname)
    data = mat73.loadmat(path)
    data1 = data[f'{var_name}_{group}']
    if var_name == 'PrecipMean':
        month_data1 = np.array([data1[:, :, i:i+11, :].sum(axis=2)
                                for i in range(0, 360, 30)])
        units = 'mm/month'
    elif var_name == 'T2mMean':
        month_data1 = np.array([data1[:, :, i:i+11, :].mean(axis=2)
                                for i in range(0, 360, 30)])
        units = 'K'
    month_data2 = []
    for year in range(0, 99):
        for month in range(0, 12):
            month_data2.append(month_data1[month, :, :, year])
    month_data2 = np.array(month_data2)
    time = pd.date_range('1/15/2001', periods=99*12, freq='M')
    mons_data = xr.Dataset({var_name: (['time', 'lon', 'lat'], month_data2)},
                           coords=dict(time=time, lon=lon, lat=lat),
                           attrs=dict(units=units, long_name=fname))
    if savefile:
        if outpath is None:
            outpath = fpath
        mons_data.to_netcdf(os.path.join(outpath, f'Climate2_99years_{var_name}_{group}.nc'))

    return mons_data


def main_Cliamte3():
    write_climate_diff2nc(file_name='Climate3.mat', var_name='Precip_diff_sce_ctl', suffix='3')
    write_climate_diff2nc(file_name='Climate3.mat', var_name='T2m_diff_sce_ctl', suffix='3')
    write_climate_diff2nc(file_name='Climate3.mat', var_name='Precip_diff_scenew_ctl', suffix='3')
    write_climate_diff2nc(file_name='Climate3.mat', var_name='T2m_diff_scenew_ctl', suffix='3')


def main_Climate99():
    pr_ctl = write_climate99_nc(var_name='PrecipMean', group='ctl')
    tm_ctl = write_climate99_nc(var_name='T2mMean', group='ctl')
    pr_sce = write_climate99_nc(var_name='PrecipMean', group='sce')
    tm_sce = write_climate99_nc(var_name='T2mMean', group='sce')