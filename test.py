# Python imports
import logging

# Libs
import geopandas as gpd
import shapely.geometry as shpg

# Locals
import oggm.cfg as cfg
from oggm import utils, workflow, tasks

# For timing the run
import time
start = time.time()

# Module logger
log = logging.getLogger(__name__)

# Initialize OGGM and set up the default run parameters
cfg.initialize(logging_level='WORKFLOW')
rgi_version = '61'
rgi_region = '11'  # Region Central Europe

# Here we override some of the default parameters
# How many grid points around the glacier?
# Make it large if you expect your glaciers to grow large:
# here, 80 is more than enough
cfg.PARAMS['border'] = 80

# Local working directory (where OGGM will write its output)
WORKING_DIR = utils.gettempdir('OGGM_Rofental')
utils.mkdir(WORKING_DIR, reset=True)
cfg.PATHS['working_dir'] = WORKING_DIR

# RGI file
path = utils.get_rgi_region_file(rgi_region, version=rgi_version)
rgidf = gpd.read_file(path)

# Get the Rofental Basin file
path = utils.get_demo_file('rofental_hydrosheds.shp')
basin = gpd.read_file(path)

# Take all glaciers in the Rofental Basin
in_bas = [basin.geometry.contains(shpg.Point(x, y))[0] for
          (x, y) in zip(rgidf.CenLon, rgidf.CenLat)]
rgidf = rgidf.loc[in_bas]

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

log.workflow('Starting OGGM run')
log.workflow('Number of glaciers: {}'.format(len(rgidf)))

# Go - get the pre-processed glacier directories
gdirs = workflow.init_glacier_directories(rgidf, from_prepro_level=4)

# We can step directly to a new experiment!
# Random climate representative for the recent climate (1985-2015)
# This is a kind of "commitment" run
workflow.execute_entity_task(tasks.run_random_climate, gdirs,
                             nyears=300, y0=2000, seed=1,
                             output_filesuffix='_commitment')
# Now we add a positive and a negative bias to the random temperature series
workflow.execute_entity_task(tasks.run_random_climate, gdirs,
                             nyears=300, y0=2000, seed=2,
                             temperature_bias=0.5,
                             output_filesuffix='_bias_p')
workflow.execute_entity_task(tasks.run_random_climate, gdirs,
                             nyears=300, y0=2000, seed=3,
                             temperature_bias=-0.5,
                             output_filesuffix='_bias_m')

# Write the compiled output
utils.compile_glacier_statistics(gdirs)
utils.compile_run_output(gdirs, input_filesuffix='_commitment')
utils.compile_run_output(gdirs, input_filesuffix='_bias_p')
utils.compile_run_output(gdirs, input_filesuffix='_bias_m')

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.workflow('OGGM is done! Time needed: %d:%02d:%02d' % (h, m, s))

# Imports
import os
import xarray as xr
import matplotlib.pyplot as plt
from oggm.utils import get_demo_file, gettempdir

# Local working directory (where OGGM wrote its output)
WORKING_DIR = gettempdir('OGGM_Rofental')

# Read the files using xarray
ds = xr.open_dataset(os.path.join(WORKING_DIR, 'run_output_commitment.nc'))
dsp = xr.open_dataset(os.path.join(WORKING_DIR, 'run_output_bias_p.nc'))
dsm = xr.open_dataset(os.path.join(WORKING_DIR, 'run_output_bias_m.nc'))

# Compute and plot the regional sums
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
# Volume
(ds.volume.sum(dim='rgi_id') * 1e-9).plot(ax=ax1, label='[1985-2015]')
(dsp.volume.sum(dim='rgi_id') * 1e-9).plot(ax=ax1, label='+0.5°C')
(dsm.volume.sum(dim='rgi_id') * 1e-9).plot(ax=ax1, label='-0.5°C')
ax1.legend(loc='best')
# Area
(ds.area.sum(dim='rgi_id') * 1e-6).plot(ax=ax2, label='[1985-2015]')
(dsp.area.sum(dim='rgi_id') * 1e-6).plot(ax=ax2, label='+0.5°C')
(dsm.area.sum(dim='rgi_id') * 1e-6).plot(ax=ax2, label='-0.5°C')
plt.tight_layout()

# Pick a specific glacier (Hintereisferner)
rid = 'RGI60-11.00897'

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
# Volume
(ds.volume.sel(rgi_id=rid) * 1e-9).plot(ax=ax1, label='[1985-2015]')
(dsp.volume.sel(rgi_id=rid) * 1e-9).plot(ax=ax1, label='+0.5°C')
(dsm.volume.sel(rgi_id=rid) * 1e-9).plot(ax=ax1, label='-0.5°C')
ax1.legend(loc='best')
# Length
(ds.length.sel(rgi_id=rid) * 1e-3).plot(ax=ax2, label='[1985-2015]')
(dsp.length.sel(rgi_id=rid) * 1e-3).plot(ax=ax2, label='+0.5°C')
(dsm.length.sel(rgi_id=rid) * 1e-3).plot(ax=ax2, label='-0.5°C')
plt.tight_layout()
plt.show()