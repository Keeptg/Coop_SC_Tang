import os
import logging
import sys
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from oggm import utils, tasks, cfg, workflow
from oggm.core.flowline import *
from oggm.core.massbalance import *
from oggm.core import gcm_climate
import geopandas as gpd


log = logging.getLogger(__name__)

@entity_task(log)
def get_clim_bias(gdir, fpath_prcp_diff, fpath_temp_diff):
    """:cvar
    """

    prcp_diff_ds = xr.open_dataset(fpath_prcp_diff)
    temp_diff_ds = xr.open_dataset(fpath_temp_diff)
    lat, lon = gdir.cenlat, gdir.cenlon
    prcp_diff = prcp_diff_ds.sel(lat=lat, lon=lon, method='nearest')
    temp_diff = temp_diff_ds.sel(lat=lat, lon=lon, method='nearest')
    gdir.prcp_diff = prcp_diff.pr.values
    gdir.temp_diff = temp_diff.tas.values


class MyPastMassBalance(MassBalanceModel):
    """Mass balance during the climate data period."""

    def __init__(self, gdir, mu_star=None, bias=None,
                 filename='climate_historical', input_filesuffix='',
                 repeat=False, ys=None, ye=None, check_calib_params=True):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mu_star : float, optional
            set to the alternative value of mu* you want to use
            (the default is to use the calibrated value).
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data.
        input_filesuffix : str
            the file suffix of the input climate file
        repeat : bool
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way
        ys : int
            The start of the climate period where the MB model is valid
            (default: the period with available data)
        ye : int
            The end of the climate period where the MB model is valid
            (default: the period with available data)
        check_calib_params : bool
            OGGM will try hard not to use wrongly calibrated mu* by checking
            the parameters used during calibration and the ones you are
            using at run time. If they don't match, it will raise an error.
            Set to False to suppress this check.

        Attributes
        ----------
        temp_bias : float, default 0
            Add a temperature bias to the time series
        prcp_bias : float, default 1
            Precipitation factor to the time series (called bias for
            consistency with `temp_bias`)
        """

        super(MyPastMassBalance, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        if mu_star is None:
            df = gdir.read_json('local_mustar')
            mu_star = df['mu_star_glacierwide']
            if check_calib_params:
                if not df['mu_star_allsame']:
                    msg = ('You seem to use the glacier-wide mu* to compute '
                           'the mass-balance although this glacier has '
                           'different mu* for its flowlines. Set '
                           '`check_calib_params=False` to prevent this '
                           'error.')
                    raise InvalidWorkflowError(msg)

        if bias is None:
            if cfg.PARAMS['use_bias_for_run']:
                df = gdir.read_json('local_mustar')
                bias = df['bias']
            else:
                bias = 0.

        self.mu_star = mu_star
        self.bias = bias

        # Parameters
        self.t_solid = cfg.PARAMS['temp_all_solid']
        self.t_liq = cfg.PARAMS['temp_all_liq']
        self.t_melt = cfg.PARAMS['temp_melt']
        prcp_fac = cfg.PARAMS['prcp_scaling_factor']
        default_grad = cfg.PARAMS['temp_default_gradient']

        # Check the climate related params to the GlacierDir to make sure
        if check_calib_params:
            mb_calib = gdir.get_climate_info()['mb_calib_params']
            for k, v in mb_calib.items():
                if v != cfg.PARAMS[k]:
                    msg = ('You seem to use different mass-balance parameters '
                           'than used for the calibration. Set '
                           '`check_calib_params=False` to ignore this '
                           'warning.')
                    raise InvalidWorkflowError(msg)

        # Public attrs
        self.hemisphere = gdir.hemisphere
        self.temp_bias = 0.
        self.prcp_bias = 0.
        self.repeat = repeat

        # Read file
        fpath = gdir.get_filepath(filename, filesuffix=input_filesuffix)
        with ncDataset(fpath, mode='r') as nc:
            # time
            time = nc.variables['time']
            time = netCDF4.num2date(time[:], time.units)
            ny, r = divmod(len(time), 12)
            if r != 0:
                raise ValueError('Climate data should be N full years')
            # This is where we switch to hydro float year format
            # Last year gives the tone of the hydro year
            self.years = np.repeat(np.arange(time[-1].year-ny+1,
                                             time[-1].year+1), 12)
            self.months = np.tile(np.arange(1, 13), ny)
            # Read timeseries
            self.temp = nc.variables['temp'][:]
            self.prcp = nc.variables['prcp'][:] * prcp_fac
            if 'gradient' in nc.variables:
                grad = nc.variables['gradient'][:]
                # Security for stuff that can happen with local gradients
                g_minmax = cfg.PARAMS['temp_local_gradient_bounds']
                grad = np.where(~np.isfinite(grad), default_grad, grad)
                grad = clip_array(grad, g_minmax[0], g_minmax[1])
            else:
                grad = self.prcp * 0 + default_grad
            self.grad = grad
            self.ref_hgt = nc.ref_hgt
            self.ys = self.years[0] if ys is None else ys
            self.ye = self.years[-1] if ye is None else ye

    def get_monthly_climate(self, heights, year=None):
        """Monthly climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """

        y, m = floatyear_to_date(year)
        if self.repeat:
            y = self.ys + (y - self.ys) % (self.ye - self.ys + 1)
        if y < self.ys or y > self.ye:
            raise ValueError('year {} out of the valid time bounds: '
                             '[{}, {}]'.format(y, self.ys, self.ye))
        pok = np.where((self.years == y) & (self.months == m))[0][0]

        # Read timeseries
        itemp = self.temp[pok] + self.temp_bias
        iprcp = np.clip(self.prcp[pok] + self.prcp_bias, 0, np.nan)
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        temp = np.ones(npix) * itemp + igrad * (heights - self.ref_hgt)
        tempformelt = temp - self.t_melt
        clip_min(tempformelt, 0, out=tempformelt)

        # Compute solid precipitation from total precipitation
        prcp = np.ones(npix) * iprcp
        fac = 1 - (temp - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol = prcp * clip_array(fac, 0, 1)

        return temp, tempformelt, prcp, prcpsol

    def _get_2d_annual_climate(self, heights, year):
        # Avoid code duplication with a getter routine
        year = np.floor(year)
        if self.repeat:
            year = self.ys + (year - self.ys) % (self.ye - self.ys + 1)
        if year < self.ys or year > self.ye:
            raise ValueError('year {} out of the valid time bounds: '
                             '[{}, {}]'.format(year, self.ys, self.ye))
        pok = np.where(self.years == year)[0]
        if len(pok) < 1:
            raise ValueError('Year {} not in record'.format(int(year)))

        # Read timeseries
        itemp = self.temp[pok] + self.temp_bias
        iprcp = np.clip(self.prcp[pok] + self.prcp_bias, 0, np.nan)
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        heights = np.asarray(heights)
        npix = len(heights)
        grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
        grad_temp *= (heights.repeat(12).reshape(grad_temp.shape) -
                      self.ref_hgt)
        temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp
        temp2dformelt = temp2d - self.t_melt
        clip_min(temp2dformelt, 0, out=temp2dformelt)

        # Compute solid precipitation from total precipitation
        prcp = np.atleast_2d(iprcp).repeat(npix, 0)
        fac = 1 - (temp2d - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol = prcp * clip_array(fac, 0, 1)

        return temp2d, temp2dformelt, prcp, prcpsol

    def get_annual_climate(self, heights, year=None):
        """Annual climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """
        t, tfmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
        return (t.mean(axis=1), tfmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_monthly_mb(self, heights, year=None, **kwargs):

        _, tmelt, _, prcpsol = self.get_monthly_climate(heights, year=year)
        mb_month = prcpsol - self.mu_star * tmelt
        mb_month -= self.bias * SEC_IN_MONTH / SEC_IN_YEAR
        return mb_month / SEC_IN_MONTH / self.rho

    def get_annual_mb(self, heights, year=None, **kwargs):

        _, temp2dformelt, _, prcpsol = self._get_2d_annual_climate(heights,
                                                                   year)
        mb_annual = np.sum(prcpsol - self.mu_star * temp2dformelt, axis=1)
        return (mb_annual - self.bias) / SEC_IN_YEAR / self.rho


class MyRandomMassBalance(MassBalanceModel):
    """Random shuffle of all MB years within a given time period.

    This is useful for finding a possible past glacier state or for sensitivity
    experiments.

    Note that this is going to be sensitive to extreme years in certain
    periods, but it is by far more physically reasonable than other
    approaches based on gaussian assumptions.
    """

    def __init__(self, gdir, mu_star=None, bias=None,
                 y0=None, halfsize=15, seed=None,
                 filename='climate_historical', input_filesuffix='',
                 all_years=False, unique_samples=False):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        mu_star : float, optional
            set to the alternative value of mu* you want to use
            (the default is to use the calibrated value)
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        y0 : int, optional, default: tstar
            the year at the center of the period of interest. The default
            is to use tstar as center.
        halfsize : int, optional
            the half-size of the time window (window size = 2 * halfsize + 1)
        seed : int, optional
            Random seed used to initialize the pseudo-random number generator.
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data.
        input_filesuffix : str
            the file suffix of the input climate file
        all_years : bool
            if True, overwrites ``y0`` and ``halfsize`` to use all available
            years.
        unique_samples: bool
            if true, chosen random mass-balance years will only be available
            once per random climate period-length
            if false, every model year will be chosen from the random climate
            period with the same probability
        """

        super(MyRandomMassBalance, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.mbmod = MyPastMassBalance(gdir, mu_star=mu_star, bias=bias,
                                       filename=filename,
                                       input_filesuffix=input_filesuffix)

        # Climate period
        if all_years:
            self.years = self.mbmod.years
        else:
            if y0 is None:
                df = gdir.read_json('local_mustar')
                y0 = df['t_star']
            self.years = np.arange(y0-halfsize, y0+halfsize+1)
        self.yr_range = (self.years[0], self.years[-1]+1)
        self.ny = len(self.years)
        self.hemisphere = gdir.hemisphere

        # RandomState
        self.rng = np.random.RandomState(seed)
        self._state_yr = dict()

        # Sampling without replacement
        self.unique_samples = unique_samples
        if self.unique_samples:
            self.sampling_years = self.years

    @property
    def temp_bias(self):
        """Temperature bias to add to the original series."""
        return self.mbmod.temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Temperature bias to add to the original series."""
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.temp_bias = value

    @property
    def prcp_bias(self):
        """Precipitation factor to apply to the original series."""
        return self.mbmod.prcp_bias

    @prcp_bias.setter
    def prcp_bias(self, value):
        """Precipitation factor to apply to the original series."""
        for attr_name in ['_lazy_interp_yr', '_lazy_interp_m']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)
        self.mbmod.prcp_bias = value

    @property
    def bias(self):
        """Residual bias to apply to the original series."""
        return self.mbmod.bias

    @bias.setter
    def bias(self, value):
        """Residual bias to apply to the original series."""
        self.mbmod.bias = value

    def get_state_yr(self, year=None):
        """For a given year, get the random year associated to it."""
        year = int(year)
        if year not in self._state_yr:
            if self.unique_samples:
                # --- Sampling without replacement ---
                if self.sampling_years.size == 0:
                    # refill sample pool when all years were picked once
                    self.sampling_years = self.years
                # choose one year which was not used in the current period
                _sample = self.rng.choice(self.sampling_years)
                # write chosen year to dictionary
                self._state_yr[year] = _sample
                # update sample pool: remove the chosen year from it
                self.sampling_years = np.delete(
                    self.sampling_years,
                    np.where(self.sampling_years == _sample))
            else:
                # --- Sampling with replacement ---
                self._state_yr[year] = self.rng.randint(*self.yr_range)
        return self._state_yr[year]

    def get_monthly_mb(self, heights, year=None, **kwargs):
        ryr, m = floatyear_to_date(year)
        ryr = date_to_floatyear(self.get_state_yr(ryr), m)
        return self.mbmod.get_monthly_mb(heights, year=ryr)

    def get_annual_mb(self, heights, year=None, **kwargs):
        ryr = self.get_state_yr(int(year))
        return self.mbmod.get_annual_mb(heights, year=ryr)


@entity_task(log)
def run_my_random_climate(gdir, fpath_prcp_diff=None, fpath_temp_diff=None, nyears=1000,
                          y0=None, halfsize=15,
                          bias=None, seed=None,
                          store_monthly_step=False,
                          climate_filename='climate_historical',
                          climate_input_filesuffix='',
                          output_filesuffix='', init_model_fls=None,
                          zero_initial_glacier=False,
                          unique_samples=False,
                          **kwargs):
    """Runs the random mass-balance model for a given number of years.

    This will initialize a
    :py:class:`oggm.core.massbalance.MultipleFlowlineMassBalance`,
    and run a :py:func:`oggm.core.flowline.robust_model_run`.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    nyears : int
        length of the simulation
    y0 : int, optional
        central year of the random climate period. The default is to be
        centred on t*.
    halfsize : int, optional
        the half-size of the time window (window size = 2 * halfsize + 1)
    bias : float
        bias of the mb model. Default is to use the calibrated one, which
        is often a better idea. For t* experiments it can be useful to set it
        to zero
    seed : int
        seed for the random generator. If you ignore this, the runs will be
        different each time. Setting it to a fixed seed accross glaciers can
        be usefull if you want to have the same climate years for all of them
    temperature_bias : float
        add a bias to the temperature timeseries
    store_monthly_step : bool
        whether to store the diagnostic data at a monthly time step or not
        (default is yearly)
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' (default) or
        'gcm_data'
    climate_input_filesuffix: str
        filesuffix for the input climate file
    output_filesuffix : str
        this add a suffix to the output file (useful to avoid overwriting
        previous experiments)
    init_model_fls : []
        list of flowlines to use to initialise the model (the default is the
        present_time_glacier file from the glacier directory)
    zero_initial_glacier : bool
        if true, the ice thickness is set to zero before the simulation
    unique_samples: bool
        if true, chosen random mass-balance years will only be available once
        per random climate period-length
        if false, every model year will be chosen from the random climate
        period with the same probability
    kwargs : dict
        kwargs to pass to the FluxBasedModel instance
    """

    mb = MultipleFlowlineMassBalance(gdir, mb_model_class=MyRandomMassBalance,
                                     y0=y0, halfsize=halfsize,
                                     bias=bias, seed=seed,
                                     filename=climate_filename,
                                     input_filesuffix=climate_input_filesuffix,
                                     unique_samples=unique_samples)

    lat, lon = gdir.cenlat, gdir.cenlon

    if fpath_prcp_diff:
        prcp_diff_ds = xr.open_dataset(fpath_prcp_diff)
        prcp_diff = prcp_diff_ds.sel(lat=lat, lon=lon, method='nearest')
        mb.prcp_bias = prcp_diff.pr.values

    if fpath_temp_diff:
        temp_diff_ds = xr.open_dataset(fpath_temp_diff)
        temp_diff = temp_diff_ds.sel(lat=lat, lon=lon, method='nearest')
        mb.temp_bias = temp_diff.tas.values

    return robust_model_run(gdir, output_filesuffix=output_filesuffix,
                            mb_model=mb, ys=0, ye=nyears,
                            store_monthly_step=store_monthly_step,
                            init_model_fls=init_model_fls,
                            zero_initial_glacier=zero_initial_glacier,
                            **kwargs)


import oggm
root_dir = '/home/users/lifei/Data/Pro_piao'
data_dir = os.path.join(root_dir, 'Data')
outpath = '/home/www/lifei/run_output/Pro_piao'
utils.mkdir(outpath)
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
rgidf_drop_dupli = rgidf.duplicated(subset='RGIId')
# Full run section 1
if np.any(rgidf_drop_dupli):
    raise ValueError("Meet duplicated glaciers")
# The End
cfg.initialize()
cfg.PARAMS['border'] = 160
cfg.PATHS['working_dir'] = utils.mkdir(os.environ["WORKDIR"])
cfg.PARAMS['continue_on_error'] = True

#gdirs = workflow.init_glacier_directories(rgidf, from_prepro_level=4,
#                                          reset=True, force=True)
gdirs = workflow.init_glacier_directories(rgidf)
# Full run section 2
task_list = [
    tasks.define_glacier_region, 
    tasks.glacier_masks, 
    tasks.compute_centerlines,
    tasks.initialize_flowlines,
    tasks.compute_downstream_line,
    tasks.compute_downstream_bedshape,
    tasks.catchment_area,
    tasks.catchment_intersections,
    tasks.catchment_width_geom,
    tasks.catchment_width_correction,
    tasks.process_cru_data, 
    tasks.local_t_star,
    tasks.mu_star_calibration,
    tasks.prepare_for_inversion,
    tasks.mass_conservation_inversion,
    tasks.filter_inversion_output,
    tasks.init_present_time_glacier
        ]

for task in task_list:
    workflow.execute_entity_task(task, gdirs)
# The End

#workflow.execute_entity_task(tasks.run_random_glimate, gdirs, y0=2000, nyears=2000)
y0 = 2000
nyears = 2000
halfsize = 15
workflow.execute_entity_task(run_my_random_climate, gdirs, nyears=nyears, y0=y0, seed=1, halfsize=halfsize,
                             output_filesuffix=f'_origin_hf{halfsize}')
for mtype in ['1', '2']:
    fpath_prcp_diff = os.path.join(data_dir, f'prec_diff{mtype}.nc')
    fpath_temp_diff = os.path.join(data_dir, f'temp_diff{mtype}.nc')
    workflow.execute_entity_task(run_my_random_climate, gdirs, nyears=nyears, y0=y0, seed=1, halfsize=halfsize,
                                 output_filesuffix=f'_exper{mtype}_hf{halfsize}',
                                 fpath_temp_diff=fpath_temp_diff,
                                 fpath_prcp_diff=fpath_prcp_diff)

output_list = []
suffixes = [f'_origin_hf{halfsize}', f'_exper1_hf{halfsize}', f'_exper2_hf{halfsize}']
for suffix in suffixes:
    output_list.append(utils.compile_run_output(gdirs, input_filesuffix=suffix, 
                                                path=os.path.join(outpath, 
                                                'result_border160'+suffix+'.nc')))

#y0 = 2000
#nyears = 2000
#halfsize = 15
#workflow.execute_entity_task(run_my_random_climate, gdirs, nyears=nyears, y0=y0, seed=1, halfsize=halfsize,
#                             output_filesuffix=f'_origin_hf{halfsize}')
#for mtype in ['1', '2']:
#    fpath_prcp_diff = os.path.join(data_dir, f'prec_diff{mtype}.nc')
#    fpath_temp_diff = os.path.join(data_dir, f'temp_diff{mtype}.nc')
#    workflow.execute_entity_task(run_my_random_climate, gdirs, nyears=nyears, y0=y0, seed=1, halfsize=halfsize,
#                                 output_filesuffix=f'_exper{mtype}_hf{halfsize}',
#                                 fpath_temp_diff=fpath_temp_diff,
#                                 fpath_prcp_diff=fpath_prcp_diff)
#
#output_list = []
#suffixes = [f'_origin_hf{halfsize}', f'_exper1_hf{halfsize}', f'_exper2_hf{halfsize}']
#for suffix in suffixes:
#
#    output_list.append(utils.compile_run_output(gdirs, input_filesuffix=suffix, 
#                                                path=os.path.join(outpath, 'result'+suffix+'.nc')))


#import matplotlib.pyplot as plt
#fig, ax = plt.subplots(1, 3)
#ylabels = ['Volume (km3)', 'Area (km2)', 'ELA (m)']
#colors = ['black', 'red', 'blue']
#for i, label in enumerate(['Origin', 'Exper1', 'Exper1']):
#    ds = output_list[i]
#    volumes = ds.isel(rgi_id=0).volume.values * 1e-9
#    areas = ds.isel(rgi_id=0).area.values * 1e-6
#    elas = ds.isel(rgi_id=0).ela.values
#    ax[0].plot(range(len(volumes)), volumes, color=colors[i])
#    ax[1].plot(range(len(areas)), areas, color=colors[i])
#    ax[2].plot(range(len(elas)), elas, color=colors[i])
#
#ax[0].set_ylabel(ylabels[0])
#ax[1].set_ylabel(ylabels[1])
#ax[2].set_ylabel(ylabels[2])