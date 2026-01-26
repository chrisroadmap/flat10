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
        "input_mode": "emissions",
        "greenhouse_gas": True,
        "aerosol_chemistry_from_emissions": False,
        "aerosol_chemistry_from_concentration": False,
    },
    "CH4": {
        "type": "ch4",
        "input_mode": "emissions",
        "greenhouse_gas": True,
        "aerosol_chemistry_from_emissions": False,
        "aerosol_chemistry_from_concentration": False,
    },
    "N2O": {
        "type": "n2o",
        "input_mode": "emissions",
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
cal_df['1.2.0'] = pd.read_csv(cal_1_2_0_file, index_col=0)

# %%
# Not yet on Zenodo
cal_df['1.4.0'] = pd.read_csv('../data/calibration/v1.4.0/calibrated_constrained_parameters.csv', index_col=0)
cal_df['1.4.1'] = pd.read_csv('../data/calibration/v1.4.1/calibrated_constrained_parameters.csv', index_col=0)

# %%
with open('../output/1pctCO2-1000PgC-airborne.pkl', 'rb') as pk:
    airborne_restarts = pickle.load(pk)

with open('../output/1pctCO2-1000PgC-alpha.pkl', 'rb') as pk:
    alpha_restarts = pickle.load(pk)

with open('../output/1pctCO2-1000PgC-cumulative.pkl', 'rb') as pk:
    cumulative_restarts = pickle.load(pk)

with open('../output/1pctCO2-1000PgC-forcing.pkl', 'rb') as pk:
    forcing_restarts = pickle.load(pk)

with open('../output/1pctCO2-1000PgC-gasbox.pkl', 'rb') as pk:
    gasbox_restarts = pickle.load(pk)

with open('../output/1pctCO2-1000PgC-temperature.pkl', 'rb') as pk:
    temperature_restarts = pickle.load(pk)

# %%
scenarios = ['1pctCO2-brch-1000PgC']

# %%
conc_df = pd.read_csv('../data/1pctCO2_concentrations.csv', index_col=0)

# %%
f = {}
for cal in calibrations:
    f[cal] = FAIR()
    f[cal].define_time(0, 200, 1)
    f[cal].define_scenarios(scenarios)
    f[cal].define_configs(list(cal_df[cal].index))
    f[cal].define_species(species, properties)
    f[cal].allocate()
    
    # fill emissions: zero
    f[cal].emissions.loc[dict(specie="CO2")] = 0
    f[cal].emissions.loc[dict(specie="CH4")] = 0
    f[cal].emissions.loc[dict(specie="N2O")] = 0
    
    # constant pre-industrial concentration for non-CO2 GHGs
    f[cal].concentration.loc[dict(specie='CH4')] = 808.2490285
    f[cal].concentration.loc[dict(specie='N2O')] = 273.021047
    
    # Get default species configs
    f[cal].fill_species_configs()
    
    # Climate response
    fill(f[cal].climate_configs['ocean_heat_capacity'], cal_df[cal].loc[:,'clim_c1':'clim_c3'])
    fill(f[cal].climate_configs['ocean_heat_transfer'], cal_df[cal].loc[:,'clim_kappa1':'clim_kappa3'])
    fill(f[cal].climate_configs['deep_ocean_efficacy'], cal_df[cal].loc[:,'clim_epsilon'])
    fill(f[cal].climate_configs['gamma_autocorrelation'], cal_df[cal].loc[:,'clim_gamma'])
    fill(f[cal].climate_configs['stochastic_run'], False)
    
    # carbon cycle
    fill(f[cal].species_configs['iirf_0'], cal_df[cal].loc[:, 'cc_r0'].values.squeeze(), specie='CO2')
    fill(f[cal].species_configs['iirf_airborne'], cal_df[cal].loc[:, 'cc_rA'].values.squeeze(), specie='CO2')
    fill(f[cal].species_configs['iirf_uptake'], cal_df[cal].loc[:, 'cc_rU'].values.squeeze(), specie='CO2')
    fill(f[cal].species_configs['iirf_temperature'], cal_df[cal].loc[:, 'cc_rT'].values.squeeze(), specie='CO2')
    
    # Scale CO2 forcing based on its 4xCO2 calibration
    fill(f[cal].species_configs["forcing_scale"], cal_df[cal]["fscale_CO2"].values.squeeze(), specie='CO2')
    
    # initial condition of CO2 concentration (but not baseline for forcing calculations)
    fill(f[cal].species_configs['baseline_concentration'], 284.3169988, specie='CO2')
    fill(f[cal].species_configs['baseline_concentration'], 808.2490285, specie='CH4')
    fill(f[cal].species_configs['baseline_concentration'], 273.021047, specie='N2O')
    
    # set initial conditions
    for iconf, config in enumerate(f[cal].configs):
        initialise(f[cal].concentration, 808.2490285, specie="CH4")
        initialise(f[cal].concentration, 273.021047, specie="N2O")
        # I forgot to save out the CO2 concentration timebound, but luckily this data can be inferred in the xarray dump.
        initialise(f[cal].concentration, conc_df.loc[airborne_restarts[cal][config].timebounds.data].values[0], specie="CO2", config=config)
        initialise(f[cal].forcing, forcing_restarts[cal][config], config=config)
        initialise(f[cal].temperature, temperature_restarts[cal][config], config=config)
        initialise(f[cal].airborne_emissions, airborne_restarts[cal][config], config=config)
        initialise(f[cal].cumulative_emissions, cumulative_restarts[cal][config], config=config)
        f[cal].gas_partitions[0, iconf, :, :] = gasbox_restarts[cal][config]
    
    f[cal].run()

# %%
for cal in calibrations:
    fig, ax = pl.subplots()
    pl.plot(f[cal].temperature.loc[dict(layer=0, scenario=scenarios[0])]);

# %%
for cal in calibrations:
    fig, ax = pl.subplots()
    pl.plot(f[cal].temperature.loc[dict(layer=0, scenario=scenarios[0])][:, :10] - f[cal].temperature.loc[dict(layer=0, scenario=scenarios[0], timebounds=0)][:10]);

# %%
for cal in calibrations:
    fig, ax = pl.subplots()
    pl.plot(f[cal].concentration.loc[dict(specie="CO2", scenario=scenarios[0])] - f[cal].concentration.loc[dict(specie="CO2", scenario=scenarios[0], timebounds=0)]);

# %%
for cal in calibrations:
    pl.hist(f[cal].temperature.loc[dict(layer=0, timebounds=0, scenario=scenarios[0])], alpha=0.4, bins=np.arange(0.8, 3.2, 0.2), density=True)

# %%
# normalise tcre to exactly 1000 PgC
tcre_norm = {}
for cal in calibrations:
    tcre_norm[cal] = 44.009/12.011 * 1000 / f[cal].cumulative_emissions.loc[dict(specie="CO2", timebounds=0, scenario=scenarios[0])]
tcre_norm[cal]

# %%
tcre_1pctCO2 = {}
for cal in calibrations:
    tcre_1pctCO2[cal] = tcre_norm[cal] * f[cal].temperature.loc[dict(layer=0, timebounds=0, scenario=scenarios[0])]

# %%
zec50_1pctCO2 = {}
for cal in calibrations:
    zec50_1pctCO2[cal] = (
        f[cal].temperature.loc[dict(layer=0, timebounds=50, scenario=scenarios[0])] -
        f[cal].temperature.loc[dict(layer=0, timebounds=0, scenario=scenarios[0])]
    )

# %%
for cal in calibrations:
    pl.hist(zec50_1pctCO2[cal], alpha=0.4, density=True, bins=np.arange(-0.4, 0.9, 0.1))

# %%
for cal in calibrations:
    print(zec50_1pctCO2[cal].mean().data)

# %%
zec100_1pctCO2 = {}
for cal in calibrations:
    zec100_1pctCO2[cal] = (
        f[cal].temperature.loc[dict(layer=0, timebounds=100, scenario=scenarios[0])] -
        f[cal].temperature.loc[dict(layer=0, timebounds=0, scenario=scenarios[0])]
    )

# %%
for cal in calibrations:
    pl.hist(zec100_1pctCO2[cal], alpha=0.4, density=True, bins=np.arange(-0.6, 1.4, 0.1))

# %%
for cal in calibrations:
    print(zec100_1pctCO2[cal].mean().data)

# %%
os.makedirs('../output/', exist_ok=True)
for cal in calibrations:
    df = pd.DataFrame(
        {
            "tcre": tcre_1pctCO2[cal],
            "zec50": zec50_1pctCO2[cal],
            "zec100": zec100_1pctCO2[cal],
        },
        index = f[cal].configs
    )
    df.to_csv(f'../output/1pctCO2_key-metrics_fair2.1.3_cal{cal}.csv')

# %%



# %%
