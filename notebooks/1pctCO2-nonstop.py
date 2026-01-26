# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.20.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 1pct runs with FaIR v2.1.3

# %%
import os
import copy
import pickle

import fair
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties
import pandas as pd
import pooch
import numpy as np
import matplotlib.pyplot as pl
from tqdm.auto import tqdm
import xarray as xr

# %%
fair.__version__

# %%
species = ["CO2", "CH4", "N2O"]
properties = {
    "CO2": {
        "type": "co2",
        "input_mode": "concentration",
        "greenhouse_gas": True,
        "aerosol_chemistry_from_emissions": False,
        "aerosol_chemistry_from_concentration": False,
    },
    "CH4": {
        "type": "ch4",
        "input_mode": "concentration",
        "greenhouse_gas": True,
        "aerosol_chemistry_from_emissions": False,
        "aerosol_chemistry_from_concentration": False,
    },
    "N2O": {
        "type": "n2o",
        "input_mode": "concentration",
        "greenhouse_gas": True,
        "aerosol_chemistry_from_emissions": False,
        "aerosol_chemistry_from_concentration": False,
    },
}

# %%
calibrations = ['1.4.0', '1.4.1']
cal_df = {}

# %%
cal_df['1.4.0'] = pd.read_csv('../data/calibration/v1.4.0/calibrated_constrained_parameters.csv', index_col=0)
cal_df['1.4.1'] = pd.read_csv('../data/calibration/v1.4.1/calibrated_constrained_parameters.csv', index_col=0)

# %%
conc_df = pd.read_csv('../data/1pctCO2_concentrations.csv', index_col=0)

# %%
emissions = {}
temperature = {}
airborne_fraction = {}

# %%
scenarios = ['1pctCO2']

# %%
# for cal in calibrations:
#     # set up a dummy fair run for the purposes of using the structure for year[0]
#     f_iter[cal][-1] = FAIR()
#     f_iter[cal][-1].define_time(-1, 0, 1)
#     f_iter[cal][-1].define_scenarios(scenarios)
#     f_iter[cal][-1].define_species(species, properties)
#     f_iter[cal][-1].define_configs(list(cal_df[cal].index))
#     f_iter[cal][-1].allocate()
#     fill(f_iter[cal][-1].forcing, 0)
#     fill(f_iter[cal][-1].temperature, 0)
#     fill(f_iter[cal][-1].airborne_emissions, 0)
#     fill(f_iter[cal][-1].cumulative_emissions, 0)
#     fill(f_iter[cal][-1].alpha_lifetime, 1)

# %%
for cal in calibrations:
    f = FAIR()
    f.define_time(0, 140, 1)
    f.define_scenarios(scenarios)
    f.define_species(species, properties)
    f.define_configs(list(cal_df[cal].index))
    f.allocate()
    
    f.concentration.loc[dict(specie='CO2')] = conc_df.values[:,None]
    f.concentration.loc[dict(specie='CH4')] = 808.2490285
    f.concentration.loc[dict(specie='N2O')] = 273.021047
        
    # Get default species configs
    f.fill_species_configs()
        
    # climate response
    fill(f.climate_configs["ocean_heat_capacity"], cal_df[cal].loc[:, "clim_c1":"clim_c3"].values)
    fill(
        f.climate_configs["ocean_heat_transfer"],
        cal_df[cal].loc[:, "clim_kappa1":"clim_kappa3"].values,
    )
    fill(f.climate_configs["deep_ocean_efficacy"], cal_df[cal].loc[:, "clim_epsilon"])
    fill(f.climate_configs["gamma_autocorrelation"], cal_df[cal].loc[:, "clim_gamma"])
    fill(f.climate_configs["sigma_eta"], cal_df[cal].loc[:, "clim_sigma_eta"])
    fill(f.climate_configs["sigma_xi"], cal_df[cal].loc[:, "clim_sigma_xi"])
    fill(f.climate_configs["seed"], cal_df[cal].loc[:, "seed"])
    fill(f.climate_configs["stochastic_run"], False)
    fill(f.climate_configs["use_seed"], True)
    fill(f.climate_configs["forcing_4co2"], cal_df[cal].loc[:, "clim_F_4xCO2"])
    
    # carbon cycle
    fill(f.species_configs["iirf_0"], cal_df[cal].loc[:,"cc_r0"], specie="CO2")
    fill(f.species_configs["iirf_airborne"], cal_df[cal].loc[:,"cc_rA"], specie="CO2")
    fill(f.species_configs["iirf_uptake"], cal_df[cal].loc[:,"cc_rU"], specie="CO2")
    fill(f.species_configs["iirf_temperature"], cal_df[cal].loc[:,"cc_rT"], specie="CO2")
    
    # forcing scaling
    fill(f.species_configs["forcing_scale"], cal_df[cal].loc[:, "fscale_CO2"], specie="CO2")
    
    # initial condition of CO2 concentration (but not baseline for forcing calculations)
    fill(
        f.species_configs["baseline_concentration"],
        cal_df[cal].loc[:,"cc_co2_concentration_1750"],
        specie="CO2",
    )
    fill(f.species_configs['baseline_concentration'], 808.2490285, specie='CH4')
    fill(f.species_configs['baseline_concentration'], 273.021047, specie='N2O')
        
    initialise(f.forcing, 0)
    initialise(f.temperature, 0)
    initialise(f.airborne_emissions, 0)
    initialise(f.cumulative_emissions, 0)
    #initialise(f_iter[cal][year].alpha_lifetime, f_iter[cal][year-1].alpha_lifetime[-1, ...])
    #f_iter[cal][year].gas_partitions=copy.deepcopy(f_iter[cal][year-1].gas_partitions)
    
    # do the run
    f.run(progress=False)
    temperature[cal] = f.temperature.sel(layer=0, scenario="1pctCO2")
    emissions[cal] = f.emissions.sel(scenario="1pctCO2", specie="CO2")
    airborne_fraction[cal] = f.airborne_fraction.sel(scenario="1pctCO2", specie="CO2")

# %%
pl.plot(emissions['1.4.0']);

# %%
pl.plot(airborne_fraction['1.4.0']);

# %%
emissions['1.4.0'].to_pandas().to_csv('../output/1pctCO2_emissions_fair2.1.3_cal1.4.0.csv')

# %%
emissions['1.4.1'].to_pandas().to_csv('../output/1pctCO2_emissions_fair2.1.3_cal1.4.1.csv')

# %%
temperature['1.4.0'].to_pandas().to_csv('../output/1pctCO2_temperature_fair2.1.3_cal1.4.0.csv')

# %%
temperature['1.4.1'].to_pandas().to_csv('../output/1pctCO2_temperature_fair2.1.3_cal1.4.1.csv')

# %%
airborne_fraction['1.4.0'].to_pandas().to_csv('../output/1pctCO2_airborne_fraction_fair2.1.3_cal1.4.0.csv')

# %%
airborne_fraction['1.4.1'].to_pandas().to_csv('../output/1pctCO2_airborne_fraction_fair2.1.3_cal1.4.1.csv')

# %%
