import os
import logging
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import oggm
from oggm import utils, tasks, cfg, workflow
import geopandas as gpd

print(oggm.__version__)

path10 = utils.get_rgi_region_file('10', '61')
path13 = utils.get_rgi_region_file('13', '61')
path14 = utils.get_rgi_region_file('14', '61')
path15 = utils.get_rgi_region_file('15', '61')
rgidf = gpd.read_file(path13).iloc[:15011, :]

cfg.initialize()
cfg.PARAMS['border'] = 160
cfg.PATHS['working_dir'] = utils.mkdir(os.environ["WORKDIR"])
cfg.PARAMS['continue_on_error'] = True

gdirs = workflow.init_glacier_directories(rgidf, from_prepro_level=4)

#workflow.execute_entity_task(tasks.run_random_climate, gdirs, y0=2000, nyears=2000)
