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
#
# We have to run iteratively, get the point at which 1000 PgC is hit, and save out the carbon and temperature states.

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
calibrations = ['1.2.0', '1.4.0', '1.4.1']
cal_df = {}

# %%
cal_1_2_0_file = pooch.retrieve(
    url = "https://zenodo.org/records/8399112/files/calibrated_constrained_parameters.csv",
    known_hash = "md5:de3b83432b9d071efdd1427ad31e9076"
)

# %%
cal_df['1.2.0'] = pd.read_csv(cal_1_2_0_file, index_col=0)

# %%
# Not yet on Zenodo
cal_df['1.4.0'] = pd.read_csv('../data/calibration/v1.4.0/calibrated_constrained_parameters.csv', index_col=0)
cal_df['1.4.1'] = pd.read_csv('../data/calibration/v1.4.1/calibrated_constrained_parameters.csv', index_col=0)

# %%
conc_df = pd.read_csv('../data/1pctCO2_concentrations.csv', index_col=0)

# %%
f_iter = {}
gasbox_restarts = {}
temperature_restarts = {}
forcing_restarts = {}
airborne_restarts = {}
cumulative_restarts = {}
alpha_restarts = {}

# %%
for cal in calibrations:
    f_iter[cal] = {}
    gasbox_restarts[cal] = {}
    temperature_restarts[cal] = {}
    forcing_restarts[cal] = {}
    airborne_restarts[cal] = {}
    cumulative_restarts[cal] = {}
    alpha_restarts[cal] = {}

# %%
hit1000_df = {}
for cal in calibrations:
    hit1000_df[cal] = pd.Series([False]*len(cal_df[cal]), index=cal_df[cal].index)
hit1000_df[cal]

# %%
scenarios = ['1pctCO2']

# %%
for cal in calibrations:
    # set up a dummy fair run for the purposes of using the structure for year[0]
    f_iter[cal][-1] = FAIR()
    f_iter[cal][-1].define_time(-1, 0, 1)
    f_iter[cal][-1].define_scenarios(scenarios)
    f_iter[cal][-1].define_species(species, properties)
    f_iter[cal][-1].define_configs(list(cal_df[cal].index))
    f_iter[cal][-1].allocate()
    fill(f_iter[cal][-1].forcing, 0)
    fill(f_iter[cal][-1].temperature, 0)
    fill(f_iter[cal][-1].airborne_emissions, 0)
    fill(f_iter[cal][-1].cumulative_emissions, 0)
    fill(f_iter[cal][-1].alpha_lifetime, 1)

# %%
for cal in calibrations:
    for year in tqdm(range(140)):
        f_iter[cal][year] = FAIR()
        f_iter[cal][year].define_time(year, year+1, 1)
        f_iter[cal][year].define_scenarios(scenarios)
        f_iter[cal][year].define_species(species, properties)
        f_iter[cal][year].define_configs(list(cal_df[cal].index))
        f_iter[cal][year].allocate()
    
        f_iter[cal][year].concentration.loc[dict(specie='CO2')] = conc_df.values[year:year+2,None]
        f_iter[cal][year].concentration.loc[dict(specie='CH4')] = 808.2490285
        f_iter[cal][year].concentration.loc[dict(specie='N2O')] = 273.021047
        
        # Get default species configs
        f_iter[cal][year].fill_species_configs()
        
        # climate response
        fill(f_iter[cal][year].climate_configs["ocean_heat_capacity"], cal_df[cal].loc[:, "clim_c1":"clim_c3"].values)
        fill(
            f_iter[cal][year].climate_configs["ocean_heat_transfer"],
            cal_df[cal].loc[:, "clim_kappa1":"clim_kappa3"].values,
        )
        fill(f_iter[cal][year].climate_configs["deep_ocean_efficacy"], cal_df[cal].loc[:, "clim_epsilon"])
        fill(f_iter[cal][year].climate_configs["gamma_autocorrelation"], cal_df[cal].loc[:, "clim_gamma"])
        fill(f_iter[cal][year].climate_configs["sigma_eta"], cal_df[cal].loc[:, "clim_sigma_eta"])
        fill(f_iter[cal][year].climate_configs["sigma_xi"], cal_df[cal].loc[:, "clim_sigma_xi"])
        fill(f_iter[cal][year].climate_configs["seed"], cal_df[cal].loc[:, "seed"])
        fill(f_iter[cal][year].climate_configs["stochastic_run"], False)
        fill(f_iter[cal][year].climate_configs["use_seed"], True)
        fill(f_iter[cal][year].climate_configs["forcing_4co2"], cal_df[cal].loc[:, "clim_F_4xCO2"])
    
        # carbon cycle
        fill(f_iter[cal][year].species_configs["iirf_0"], cal_df[cal].loc[:,"cc_r0"], specie="CO2")
        fill(
            f_iter[cal][year].species_configs["iirf_airborne"], cal_df[cal].loc[:,"cc_rA"], specie="CO2"
        )
        fill(f_iter[cal][year].species_configs["iirf_uptake"], cal_df[cal].loc[:,"cc_rU"], specie="CO2")
        fill(
            f_iter[cal][year].species_configs["iirf_temperature"],
            cal_df[cal].loc[:,"cc_rT"],
            specie="CO2",
        )
    
        # forcing scaling
        fill(
            f_iter[cal][year].species_configs["forcing_scale"],
            cal_df[cal].loc[:, "fscale_CO2"],
            specie="CO2",
        )
    
        # initial condition of CO2 concentration (but not baseline for forcing calculations)
        fill(
            f_iter[cal][year].species_configs["baseline_concentration"],
            cal_df[cal].loc[:,"cc_co2_concentration_1750"],
            specie="CO2",
        )
        fill(f_iter[cal][year].species_configs['baseline_concentration'], 808.2490285, specie='CH4')
        fill(f_iter[cal][year].species_configs['baseline_concentration'], 273.021047, specie='N2O')
        
        initialise(f_iter[cal][year].forcing, f_iter[cal][year-1].forcing[-1, ...])
        initialise(f_iter[cal][year].temperature, f_iter[cal][year-1].temperature[-1, ...])
        initialise(f_iter[cal][year].airborne_emissions, f_iter[cal][year-1].airborne_emissions[-1, ...])
        initialise(f_iter[cal][year].cumulative_emissions, f_iter[cal][year-1].cumulative_emissions[-1, ...])
        initialise(f_iter[cal][year].alpha_lifetime, f_iter[cal][year-1].alpha_lifetime[-1, ...])
        f_iter[cal][year].gas_partitions=copy.deepcopy(f_iter[cal][year-1].gas_partitions)
    
        # do the run
        f_iter[cal][year].run(progress=False)
    
        # check if over 1000 GtC
        for iconf, config in enumerate(f_iter[cal][year].configs):
            if not hit1000_df[cal].loc[config] and (f_iter[cal][year].cumulative_emissions[-1, 0, iconf, 0] >= 1000*44.009/12.011):
                hit1000_df[cal].loc[config] = True
                gasbox_restarts[cal][config] = copy.deepcopy(f_iter[cal][year].gas_partitions[0, iconf, :, :])
                forcing_restarts[cal][config] = copy.deepcopy(f_iter[cal][year].forcing[-1, 0, iconf, :])
                temperature_restarts[cal][config] = copy.deepcopy(f_iter[cal][year].temperature[-1, 0, iconf, :])
                airborne_restarts[cal][config] = copy.deepcopy(f_iter[cal][year].airborne_emissions[-1, 0, iconf, :])
                cumulative_restarts[cal][config] = copy.deepcopy(f_iter[cal][year].cumulative_emissions[-1, 0, iconf, :])
                alpha_restarts[cal][config] = copy.deepcopy(f_iter[cal][year].alpha_lifetime[-1, 0, iconf, :])

# %%
f_iter['cal-1.2.1']

# %%
os.makedirs('../output/', exist_ok=True)

# %%
with open('../output/1pctCO2-1000PgC-alpha.pkl', 'wb') as pk:
    pickle.dump(alpha_restarts, pk, pickle.HIGHEST_PROTOCOL)

with open('../output/1pctCO2-1000PgC-airborne.pkl', 'wb') as pk:
    pickle.dump(airborne_restarts, pk, pickle.HIGHEST_PROTOCOL)

with open('../output/1pctCO2-1000PgC-cumulative.pkl', 'wb') as pk:
    pickle.dump(cumulative_restarts, pk, pickle.HIGHEST_PROTOCOL)

with open('../output/1pctCO2-1000PgC-forcing.pkl', 'wb') as pk:
    pickle.dump(forcing_restarts, pk, pickle.HIGHEST_PROTOCOL)

with open('../output/1pctCO2-1000PgC-temperature.pkl', 'wb') as pk:
    pickle.dump(temperature_restarts, pk, pickle.HIGHEST_PROTOCOL)

with open('../output/1pctCO2-1000PgC-gasbox.pkl', 'wb') as pk:
    pickle.dump(gasbox_restarts, pk, pickle.HIGHEST_PROTOCOL)

# %%
