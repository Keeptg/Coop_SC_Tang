import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

root_dir = '/home/keeptg/Data/Study_in_Innsbruck/Pro_piao/'
data_dir = os.path.join(root_dir, 'cluster_output')
os.path.exists(data_dir)

exp1 = os.path.join(data_dir, 'result_border160_exper1_hf0.nc')
exp2 = os.path.join(data_dir, 'result_border160_exper2_hf0.nc')
orig = os.path.join(data_dir, 'result_border160_origin_hf0.nc')

ds0 = xr.open_dataset(orig)
ds1 = xr.open_dataset(exp1)
ds2 = xr.open_dataset(exp2)

vol0 = ds0.volume.sum(dim='rgi_id')
area0 = ds0.area.sum(dim='rgi_id')
vol1 = ds1.volume.sum(dim='rgi_id')
area1 = ds1.area.sum(dim='rgi_id')
vol2 = ds2.volume.sum(dim='rgi_id')
area2 = ds2.area.sum(dim='rgi_id')

fig, axs = plt.subplots(1, 2)
colors = ['black', 'red', 'blue']
ylabels = ['Volume (km$^3$)', 'Area (km$^2$)']
xlabels = 'Years'
labels = ['Original Clim.', 'Experiment Diff. 1', 'Experiment Diff. 2']

for i, (vol, area) in enumerate(zip([vol0, vol1, vol2], [area0, area1, area2])):
    label = labels[i]
    color = colors[i]
    axs[0].plot(vol.data*1e-9, color=color, label=label)
    axs[1].plot(area.data*1e-6, color=color, label=label)
    axs[0].set_ylabel(ylabels[0]), axs[0].set_xlabel(xlabels)
    axs[1].set_ylabel(ylabels[1]), axs[1].set_xlabel(xlabels)

rgi10 = ds0.rgi_id.str.contains('RGI60-13.')
ds0_r10 = ds0.isel(rgi_id=np.where(rgi10.values)[0])
ds1_r10 = ds1.isel(rgi_id=np.where(rgi10.values)[0])
ds2_r10 = ds2.isel(rgi_id=np.where(rgi10.values)[0])

vol0 = ds0_r10.volume.sum(dim='rgi_id')
area0 = ds0_r10.area.sum(dim='rgi_id')
vol1 = ds1_r10.volume.sum(dim='rgi_id')
area1 = ds1_r10.area.sum(dim='rgi_id')
vol2 = ds2_r10.volume.sum(dim='rgi_id')
area2 = ds2_r10.area.sum(dim='rgi_id')

fig, axs = plt.subplots(1, 2)
colors = ['black', 'red', 'blue']
ylabels = ['Volume (km$^3$)', 'Area (km$^2$)']
xlabels = 'Years'
labels = ['Original Clim.', 'Experiment Diff. 1', 'Experiment Diff. 2']

for i, (vol, area) in enumerate(zip([vol0, vol1, vol2], [area0, area1, area2])):
    label = labels[i]
    color = colors[i]
    axs[0].plot(vol.data*1e-9, color=color, label=label)
    axs[1].plot(area.data*1e-6, color=color, label=label)
    axs[0].set_ylabel(ylabels[0]), axs[0].set_xlabel(xlabels)
    axs[1].set_ylabel(ylabels[1]), axs[1].set_xlabel(xlabels)
axs[0].legend()
axs[1].legend()

diff_dir = os.path.join(root_dir, 'Data')
precp1 = xr.open_dataset(os.path.join(diff_dir, 'prec_diff1.nc'))
precp2 = xr.open_dataset(os.path.join(diff_dir, 'prec_diff2.nc'))
temp1 = xr.open_dataset(os.path.join(diff_dir, 'temp_diff1.nc'))
temp2 = xr.open_dataset(os.path.join(diff_dir, 'temp_diff2.nc'))

from oggm import utils, cfg
import geopandas as gpd
cfg.initialize()
path = utils.get_rgi_region_file(region='10')
rgidf = gpd.read_file(path)
print(rgidf.columns.values)
rgidf = rgidf[rgidf.O2Region.str.contains('4')]
lons = rgidf.CenLon
lats = rgidf.CenLat

precp1_ch = precp1.sel(lat=lats, lon=lons, method='nearest')
precp2_ch = precp2.sel(lat=lats, lon=lons, method='nearest')
temp1_ch = temp1.sel(lat=lats, lon=lons, method='nearest')
temp2_ch = temp2.sel(lat=lats, lon=lons, method='nearest')
precp1_arr = precp1_ch.pr.values.mean(axis=(1, 2))
precp2_arr = precp2_ch.pr.values.mean(axis=(1, 2))
temp1_arr = temp1_ch.tas.values.mean(axis=(1, 2))
temp2_arr = temp2_ch.tas.values.mean(axis=(1, 2))

fig, ax = plt.subplots()
ax.bar(np.arange(1, 13, 1)-.2, precp1_arr, width=.4, fc='none', ec='blue', label='Experiment 1')
ax.bar(np.arange(1, 13, 1)+.2, precp2_arr, width=.4, fc='none', ec='red', label='Experiment 1')
axt = ax.twinx()
axt.plot(temp1_arr, color='blue', label='Experiment 1')
axt.plot(temp2_arr, color='red', label='Experiment 2')
