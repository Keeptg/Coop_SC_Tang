# This is a sample Python script.

import os
import mat73
from scipy.io import loadmat
import numpy as np
import pandas as pd
import xarray as xr
import salem

from path_config import *


def write_mat2nc(mat_tuple, varname, years, lon_in_dim, lat_in_dim, year_in_dim, day_in_dim, month_in_dim,
                 units=None, y0=None, multiple_years_data=False):
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
        if not multiple_years_data:
            month_data = np.array([day_data[:, :, i:i + 30].sum(axis=day_in_dim)/convert_fact 
                          for i in range(0, day_data.shape[day_in_dim], 30)])
        else:
            month_data = np.array([day_data[:, :, i:i + 30, :].sum(axis=day_in_dim)/convert_fact
                          for i in range(0, day_data.shape[day_in_dim], 30)])
        long_name = 'Precipitation'
    elif data_vars == 'tas':
        if units == None:
            units = 'K'
            convert_fact = 0
        elif units == 'C':
            convert_fact = 273.15
        else:
            raise NameError(f'No units named as {units}')
        if not multiple_years_data:
            month_data = np.array([day_data[:, :, i:i + 30].mean(axis=day_in_dim) + convert_fact
                          for i in range(0, day_data.shape[day_in_dim], 30)])
        else:
            month_data = np.array([day_data[:, :, i:i + 30, :].mean(axis=day_in_dim) + convert_fact
                          for i in range(0, day_data.shape[day_in_dim], 30)])
        long_name = 'Near-Surface Air Temperature (2m)'
    else:
        raise NameError(f"No variable named as {varname}")
    num = 0
    while num < years:
        m_data = month_data if not multiple_years_data else month_data[:, :, :, num]
        try:
            ys_data = np.append(ys_data, m_data, axis=0)
        except NameError:
            ys_data = m_data
        num += 1
    if y0 is None:
        y0 = '1/15/1950'
    time = pd.date_range(y0, periods=years * 12, freq='M')
    lon = mat_tuple['lon']
    lat = mat_tuple['lat']
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
    

def write_climate3_data():
    write_climate_diff2nc(file_name='Climate3.mat', var_name='Precip_diff_sce_ctl', suffix='3')
    write_climate_diff2nc(file_name='Climate3.mat', var_name='T2m_diff_sce_ctl', suffix='3')
    write_climate_diff2nc(file_name='Climate3.mat', var_name='Precip_diff_scenew_ctl', suffix='3')
    write_climate_diff2nc(file_name='Climate3.mat', var_name='T2m_diff_scenew_ctl', suffix='3')


def _write_climate1_data(fname, varname, output_dir):

    in_path = os.path.join(root_dir, 'Climate1_99years')
    lon_lat = mat73.loadmat(os.path.join(in_path, 'Climate1_99years_latlon.mat'))
    lat = lon_lat['lat']
    lon = lon_lat['lon']
    data = mat73.loadmat(os.path.join(in_path, fname))
    data['lon'] = lon
    data['lat'] = lat
    ds = write_mat2nc(data, varname=varname, y0='1/15/2000', years=99, lon_in_dim=0, lat_in_dim=1,
                      day_in_dim=2, year_in_dim=3, month_in_dim=None, multiple_years_data=True)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if 'Precip' in varname:
        sname = 'pr'
    elif 'T2m' in varname:
        sname = 'tas'
    da = ds[sname]
    da.attrs = ds.attrs
    ds[sname] = da
    ds.to_netcdf(os.path.join(output_dir, varname+'.nc'))
    print(f"Finished {varname}.nc writing!")


def write_climate1_data():
    out_dir = os.path.join(data_dir, 'Climate1_99years')
    _write_climate1_data(fname='Climate1_99years_PrecipMean_ctl.mat', varname='PrecipMean_ctl', output_dir=out_dir)
    _write_climate1_data(fname='Climate1_99years_PrecipMean_sce.mat', varname='PrecipMean_sce', output_dir=out_dir)
    _write_climate1_data(fname='Climate1_99years_T2mMean_ctl.mat', varname='T2mMean_ctl', output_dir=out_dir)
    _write_climate1_data(fname='Climate1_99years_T2mMean_sce.mat', varname='T2mMean_sce', output_dir=out_dir)
    prcp_ctl = xr.open_dataset(os.path.join(data_dir, 'Climate1_99years', 'PrecipMean_ctl.nc'))
    prcp_sce = xr.open_dataset(os.path.join(data_dir, 'Climate1_99years', 'PrecipMean_sce.nc'))
    temp_ctl = xr.open_dataset(os.path.join(data_dir, 'Climate1_99years', 'T2mMean_ctl.nc'))
    temp_sce = xr.open_dataset(os.path.join(data_dir, 'Climate1_99years', 'T2mMean_sce.nc'))

    prcp_sce_value = prcp_sce.pr.values
    prcp_ctl_value = prcp_ctl.pr.values
    temp_sce_value = temp_sce.tas.values
    temp_ctl_value = temp_ctl.tas.values
    prcp_sce_value[:156, :, :] = prcp_ctl_value[:156, :, :]
    temp_sce_value[:156, :, :] = temp_ctl_value[:156, :, :]
    prcp_sce['pr'].values = prcp_sce_value
    temp_sce['tas'].values = temp_sce_value
    prcp_sce.to_netcdf(os.path.join(data_dir, 'Climate1_99years', 'PrecipMean_sce_ctl_2000-2010.nc'))
    temp_sce.to_netcdf(os.path.join(data_dir, 'Climate1_99years', 'T2mMean_sce_ctl_2000-2010.nc'))

    # read result for check
    prcp_sce_ctl = xr.open_dataset(os.path.join(data_dir, 'Climate1_99years', 'PrecipMean_sce_ctl_2000-2010.nc'))
    temp_sce_ctl = xr.open_dataset(os.path.join(data_dir, 'Climate1_99years', 'T2mMean_sce_ctl_2000-2010.nc'))


def _write_climate2_data(fname, varname, output_dir):

    in_path = os.path.join(root_dir, 'Data', 'Climate2_99years')
    lon_lat = mat73.loadmat(os.path.join(in_path, 'Climate2_99years_latlon.mat'))
    lat = lon_lat['lat']
    lon = lon_lat['lon']
    data = mat73.loadmat(os.path.join(in_path, fname))
    data['lon'] = lon
    data['lat'] = lat
    ds = write_mat2nc(data, varname=varname, y0='1/15/2001', years=99, lon_in_dim=0, lat_in_dim=1,
                      day_in_dim=2, year_in_dim=3, month_in_dim=None, multiple_years_data=True)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if 'Precip' in varname:
        sname = 'pr'
    elif 'T2m' in varname:
        sname = 'tas'
    da = ds[sname]
    da.attrs = ds.attrs
    ds[sname] = da
    ds.to_netcdf(os.path.join(output_dir, varname+'.nc'))
    print(f"Finished {varname}.nc writing!")


def write_climate2_data():
    out_dir = os.path.join(data_dir, 'Climate2_99years')
    _write_climate2_data(fname='Climate2_99years_PrecipMean_ctl.mat', varname='PrecipMean_ctl', output_dir=out_dir)
    _write_climate2_data(fname='Climate2_99years_PrecipMean_sce.mat', varname='PrecipMean_sce', output_dir=out_dir)
    _write_climate2_data(fname='Climate2_99years_T2mMean_ctl.mat', varname='T2mMean_ctl', output_dir=out_dir)
    _write_climate2_data(fname='Climate2_99years_T2mMean_sce.mat', varname='T2mMean_sce', output_dir=out_dir)


if __name__ == "__main__":
    #write_climate1_data()
    write_climate2_data()