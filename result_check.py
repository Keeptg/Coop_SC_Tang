import os
import numpy as np
import pandas as pd
import xarray as xr

from path_config import *
%matplotlib widget
import matplotlib.pyplot as plt

data_dir = os.path.join(root_dir, 'Climate1_2')
org_ds = xr.open_dataset(os.path.join(data_dir, 'result_origin_hf0.nc')) 
ex1_ds = xr.open_dataset(os.path.join(data_dir, 'result_exper_1_hf0.nc'))
ex2_ds = xr.open_dataset(os.path.join(data_dir, 'result_exper_2_hf0.nc'))
org_vol = org_ds.volume.sum(dim='rgi_id')
ex1_vol = ex1_ds.volume.sum(dim='rgi_id')
ex2_vol = ex2_ds.volume.sum(dim='rgi_id')

data_dir = os.path.join(root_dir, 'Climate1_2_pre')
org0_ds = xr.open_dataset(os.path.join(data_dir, 'result_border160_origin_hf0.nc')) 
ex10_ds = xr.open_dataset(os.path.join(data_dir, 'result_border160_exper1_hf0.nc'))
ex20_ds = xr.open_dataset(os.path.join(data_dir, 'result_border160_exper2_hf0.nc'))
org0_vol = org0_ds.volume.sum(dim='rgi_id')
ex10_vol = ex10_ds.volume.sum(dim='rgi_id')
ex20_vol = ex20_ds.volume.sum(dim='rgi_id')

fig, ax = plt.subplots()
ax.plot(org0_vol.hydro_year, org0_vol, color='black', label='org', ls=(0, (3, 1)))
ax.plot(ex10_vol.hydro_year, ex10_vol, color='blue', label='org', ls=(0, (3, 1)))
ax.plot(ex20_vol.hydro_year, ex20_vol, color='red', label='org', ls=(0, (3, 1)))
ax.plot(org_vol.hydro_year, org_vol, color='black', label='org')
ax.plot(ex1_vol.hydro_year, ex1_vol, color='blue', label='org')
ax.plot(ex2_vol.hydro_year, ex2_vol, color='red', label='org')

fig, ax = plt.subplots()

ax.plot(org_vol.hydro_year, org_vol, color='black', label='org')
ax.plot(ex1_vol.hydro_year, ex1_vol, color='blue', label='exp1')
ax.plot(ex2_vol.hydro_year, ex2_vol, color='red', label='exp2')
ax.set_xlabel('Year'), ax.set_ylabel('Ice volume (km$^3$)')

in_rgn10 = ['RGI60-10' in id_ for id_ in org_ds.rgi_id.values]
org_vol10 = org_ds.sel(rgi_id=in_rgn10).volume.sum(dim='rgi_id')
ex1_vol10 = ex1_ds.sel(rgi_id=in_rgn10).volume.sum(dim='rgi_id')
ex2_vol10 = ex2_ds.sel(rgi_id=in_rgn10).volume.sum(dim='rgi_id')
fig, ax = plt.subplots()
ax.plot(org_vol10.hydro_year, org_vol10, color='black', label='org')
ax.plot(ex1_vol10.hydro_year, ex1_vol10, color='blue', label='exp1')
ax.plot(ex2_vol10.hydro_year, ex2_vol10, color='red', label='exp2')
ax.set_xlabel('Year'), ax.set_ylabel('Ice volume (km$^3$)')


