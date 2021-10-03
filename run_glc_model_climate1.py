import os, pickle
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from oggm import utils, tasks, cfg, workflow
from oggm.workflow import execute_entity_task
from oggm.shop import gcm_climate
from oggm.core.flowline import *
from oggm.core.massbalance import *
import geopandas as gpd
from path_config import *


def pre_process_tasks(run_for_test=False, reset=True):

    path10 = utils.get_rgi_region_file('10', '61')
    path13 = utils.get_rgi_region_file('13', '61')
    path14 = utils.get_rgi_region_file('14', '61')
    path15 = utils.get_rgi_region_file('15', '61')
    rgidf10 = gpd.read_file(path10)
    rgidf10 = rgidf10[rgidf10.O2Region == '4']
    rgidf13 = gpd.read_file(path13)
    rgidf14 = gpd.read_file(path14)
    rgidf15 = gpd.read_file(path15)
    rgidf = pd.concat([rgidf10, rgidf13, rgidf14, rgidf15])
    if (not run_in_cluster) or run_for_test:
        rgidf = rgidf10.iloc[0:5, :]
    cfg.initialize()
    cfg.PARAMS['border'] = 160
    cfg.PATHS['working_dir'] = utils.mkdir(working_dir)
    if run_for_test == True:
        cfg.PARAMS['continue_on_error'] = False
        cfg.PARAMS['use_multiprocessing'] = False
    else:
        cfg.PARAMS['continue_on_error'] = True
        cfg.PARAMS['use_multiprocessing'] = True

    gdirs = workflow.init_glacier_directories(rgidf, from_prepro_level=5, reset=reset, force=reset)
    with open(os.path.join(cfg.PATHS['working_dir'], 'gdirs.pkl'), 'wb') as f:
        pickle.dump(gdirs, f)

    return gdirs


def run_with_job_array(climate_suffix, gdirs=None, run_for_test=False,
                       output_dir=None):
    """
    Parameters
    ----------
    climate_suffix : str, should be one of ['ctl', 'sce', 'sce_ctl_2000-2010']
    """

    if output_dir is None:
        outpath = utils.mkdir(os.path.join(cluster_dir, 'cluster_output', 'Climate_1'), reset=False)
    else:
        outpath = utils.mkdir(os.path.join(cluster_dir, output_dir),
                              reset=False)
    if gdirs is None:
        gdirs = pre_process_tasks(run_for_test=run_for_test)

    prcp_fpath = os.path.join(data_dir, 'Climate1_99years', 'PrecipMean_'+climate_suffix+'.nc')
    temp_fpath = os.path.join(data_dir, 'Climate1_99years', 'T2mMean_'+climate_suffix+'.nc')
    execute_entity_task(gcm_climate.process_cmip_data, gdirs, filesuffix='_'+climate_suffix,
                        fpath_precip=prcp_fpath, fpath_temp=temp_fpath, year_range=('2001', '2012')) 
    execute_entity_task(tasks.run_from_climate_data, gdirs, ys=2020, 
                        climate_filename='gcm_data', climate_input_filesuffix='_'+climate_suffix,
                        init_model_filesuffix='_historical', output_filesuffix=f'_transient_{climate_suffix}')
    execute_entity_task(tasks.run_random_climate, gdirs, nyears=2000, y0=2059, halfsize=39, seed=1,
                        climate_filename='gcm_data', climate_input_filesuffix=f'_'+climate_suffix,
                        output_filesuffix=f'_constant_{climate_suffix}')
    ds_t = utils.compile_run_output(gdirs, input_filesuffix=f'_transient_{climate_suffix}',
                                    path=os.path.join(outpath, f'Transient_{climate_suffix}.nc'))
    ds_c = utils.compile_run_output(gdirs, input_filesuffix=f'_constant_{climate_suffix}',
                                    path=os.path.join(outpath, f'Constant_{climate_suffix}.nc'))
    if not os.path.exists(os.path.join(outpath, 'Historical.nc')):
        ds_h = utils.compile_run_output(gdirs, input_filesuffix='_historical',
                                        path=os.path.join(outpath, 'Historical.nc'))


def plot_for_check_the_test_result():
    import matplotlib.pyplot as plt
    path = os.path.join(cluster_dir, 'cluster_output', 'Climate_1')
    c_ctl = xr.open_dataset(os.path.join(path, 'Constant_ctl.nc'))
    t_ctl = xr.open_dataset(os.path.join(path, 'Transient_ctl.nc'))
    c_sce = xr.open_dataset(os.path.join(path, 'Constant_sce.nc'))
    t_sce = xr.open_dataset(os.path.join(path, 'Transient_sce.nc'))
    c_sce_ctl = xr.open_dataset(os.path.join(path, 'Constant_sce_ctl_2000-2010.nc'))
    t_sce_ctl = xr.open_dataset(os.path.join(path, 'Transient_sce_ctl_2000-2010.nc'))
    c_ctl = c_ctl.sum(dim='rgi_id')
    c_sce = c_sce.sum(dim='rgi_id')
    c_sce_ctl = c_sce_ctl.sum(dim='rgi_id')
    fig, ax = plt.subplots()
    ax.plot(c_ctl.time, c_ctl.volume, color='k')
    ax.plot(c_sce.time, c_sce.volume, color='b')
    ax.plot(c_sce_ctl.time, c_sce_ctl.volume, color='r')

    t_ctl = t_ctl.sum(dim='rgi_id')
    t_sce = t_sce.sum(dim='rgi_id')
    t_sce_ctl = t_sce_ctl.sum(dim='rgi_id')
    fig, ax = plt.subplots()
    ax.plot(t_ctl.time, t_ctl.volume, color='k')
    ax.plot(t_sce.time, t_sce.volume, color='b')
    ax.plot(t_sce_ctl.time, t_sce_ctl.volume, color='r')

    with open(os.path.join(cfg.PATHS['working_dir'], 'gdirs.pkl'), 'rb') as f:
        gdirs = pickle.load(f)
    gdir = gdirs[0]
    os.chdir(gdir.dir)
    path = os.path.join(gdir.dir, 'model_diagnostics_constant_sce_ctl_2000-2010.nc')
    os.path.exists(path)
    ds = xr.open_dataset(path)

    ds_ctl = xr.open_dataset(os.path.join(gdir.dir, 'gcm_data_ctl.nc'))
    ds_sce = xr.open_dataset(os.path.join(gdir.dir, 'gcm_data_sce.nc'))
    ds_ctl_sce = xr.open_dataset(os.path.join(gdir.dir, 'gcm_data_sce_ctl_2000-2010.nc'))
    ds_ctl.prcp.values
    ds_sce.prcp.values
    ds_ctl_sce.prcp.values


global run_for_test
run_for_test = True

# Parameters for the combined climate run
kwargs0 = dict(climate_suffix='ctl', run_for_test=run_for_test)
kwargs1 = dict(climate_suffix='sce', run_for_test=run_for_test)
kwargs2 = dict(climate_suffix='sce_ctl_2000-2010', run_for_test=run_for_test)

kwargs_list = [kwargs0, kwargs1, kwargs2]

if not run_in_cluster:
    gdirs = pre_process_tasks(run_for_test)
    for kwargs in kwargs_list:
        run_with_job_array(gdirs=gdirs, **kwargs)
else:
    task_num = int(os.environ.get('TASK_ID'))
    run_with_job_array(**kwargs_list[task_num])