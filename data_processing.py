# This is a sample Python script.

import os
import sys
import glob
import mat73
from scipy.io import loadmat
import numpy as np
import pandas as pd
import xarray as xr

import salem


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


def write_climate2nc(years, **kwargs):

    for mtype in ['1', '2']:
        for etype in ['sce', 'ctl']:
            climate_path = os.path.join(data_dir, f'Climate{mtype}.mat')
            data = mat73.loadmat(climate_path)
            prec_ds = write_mat2nc(data, f'PrecipMean_{etype}', years, **kwargs)
            temp_ds = write_mat2nc(data, f'T2mMean_{etype}', years, **kwargs)
            prec_ds.to_netcdf(os.path.join(data_dir, f'prec{years}_{etype}_ds{mtype}.nc'))
            temp_ds.to_netcdf(os.path.join(data_dir, f'temp{years}_{etype}_ds{mtype}.nc'))


def write_climate_diff2nc(prcp_units=None, temp_units=None):

    for mtype in ['_diff1', '_diff2']:
            climate_path = os.path.join(data_dir, f'Climate{mtype}.mat')
            data = mat73.loadmat(climate_path)
            prec_ds = write_mat2nc(data, 'Precip_diff', 1, units=prcp_units)
            temp_ds = write_mat2nc(data, 'T2m_diff', 1, units=temp_units)
            prec_ds.to_netcdf(os.path.join(data_dir, f'prec{mtype}.nc'))
            temp_ds.to_netcdf(os.path.join(data_dir, f'temp{mtype}.nc'))


def main():
    global root_dir, data_dir
    root_dir = '/home/keeptg/Data/Study_in_Innsbruck/Pro_piao'
    data_dir = os.path.join(root_dir, 'Data')
    script_dir = os.path.join(root_dir, 'Script')

    write_climate_diff2nc(prcp_units='mm')

if __name__ == "__main__":
    main()