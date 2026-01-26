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
# # Flat10 runs with FaIR v2.1.3

# %%
import os
import fair
from fair import FAIR
from fair.interface import fill, initialise
from fair.io import read_properties
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import pooch
import seaborn as sns
from tqdm.auto import tqdm
import xarray as xr

# %%
fair.__version__

# %%
scenarios = ['esm-flat10', 'esm-flat10_zec', 'esm-flat10_cdr']

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
cal_df['1.2.0']

# %%
# Not yet on Zenodo
cal_df['1.4.0'] = pd.read_csv('../data/calibration/v1.4.0/calibrated_constrained_parameters.csv', index_col=0)
cal_df['1.4.1'] = pd.read_csv('../data/calibration/v1.4.1/calibrated_constrained_parameters.csv', index_col=0)

# %%
species = ['CO2', 'CH4', 'N2O']
properties = {
    "CO2": {
        'type': 'co2',
        'input_mode': 'emissions',
        'greenhouse_gas': True,
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': False
    },
    "CH4": {
        'type': 'ch4',
        'input_mode': 'emissions',
        'greenhouse_gas': True,
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': False
    },
    "N2O": {
        'type': 'n2o',
        'input_mode': 'emissions',
        'greenhouse_gas': True,
        'aerosol_chemistry_from_emissions': False,
        'aerosol_chemistry_from_concentration': False
    }
}

# %%
f = {}

# %%
for cal in calibrations:
    f[cal] = FAIR()
    f[cal].define_time(0, 320, 1)
    f[cal].define_scenarios(scenarios)
    f[cal].define_configs(list(cal_df[cal].index))
    
    # declare species and properties
    f[cal].define_species(species, properties)
    
    f[cal].allocate()
    
    # fill emissions: zero for non-CO2
    f[cal].emissions.loc[dict(specie="CH4")] = 0
    f[cal].emissions.loc[dict(specie="N2O")] = 0
    
    # constant pre-industrial concentration for non-CO2 GHGs
    f[cal].concentration.loc[dict(specie='CH4')] = 808.2490285
    f[cal].concentration.loc[dict(specie='N2O')] = 273.021047
    
    # fill emissions of CO2 for each scenario
    f[cal].emissions.loc[dict(specie="CO2", scenario="esm-flat10")] = 10 * 44.009 / 12.011
    f[cal].emissions.loc[dict(specie="CO2", scenario="esm-flat10_zec", timepoints=np.arange(0.5, 100))] = 10 * 44.009 / 12.011
    f[cal].emissions.loc[dict(specie="CO2", scenario="esm-flat10_zec", timepoints=np.arange(100.5, 320))] = 0
    f[cal].emissions.loc[dict(specie="CO2", scenario="esm-flat10_cdr", timepoints=np.arange(0.5, 100))] = 10 * 44.009 / 12.011
    f[cal].emissions.loc[dict(specie="CO2", scenario="esm-flat10_cdr", timepoints=np.arange(100.5, 200))] = np.linspace(9.9, -9.9, 100)[:, None] * 44.009 / 12.011
    f[cal].emissions.loc[dict(specie="CO2", scenario="esm-flat10_cdr", timepoints=np.arange(200.5, 300))] = -10 * 44.009 / 12.011
    f[cal].emissions.loc[dict(specie="CO2", scenario="esm-flat10_cdr", timepoints=np.arange(300.5, 320))] = 0
    
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
    initialise(f[cal].concentration, f[cal].species_configs['baseline_concentration'])
    initialise(f[cal].forcing, 0)
    initialise(f[cal].temperature, 0)
    initialise(f[cal].airborne_emissions, 0)
    initialise(f[cal].cumulative_emissions, 0)
    
    f[cal].run()

# %%
for cal in calibrations:
    fig, ax = pl.subplots(2, 2)
    ax[0,0].plot(f[cal].timepoints[:150], f[cal].emissions.loc[dict(specie="CO2", scenario="esm-flat10", timepoints=np.arange(0.5, 150))], color='k', alpha=0.1);
    ax[0,0].plot(f[cal].timepoints, f[cal].emissions.loc[dict(specie="CO2", scenario="esm-flat10_zec")], color='b', alpha=0.1);
    ax[0,0].plot(f[cal].timepoints, f[cal].emissions.loc[dict(specie="CO2", scenario="esm-flat10_cdr")], color='r', alpha=0.1, ls=':');
    ax[0,1].plot(np.arange(0, 151), f[cal].cumulative_emissions.loc[dict(specie="CO2", scenario="esm-flat10", timebounds=np.arange(0, 151))], color='k', alpha=0.1);
    ax[0,1].plot(f[cal].cumulative_emissions.loc[dict(specie="CO2", scenario="esm-flat10_zec")], color='b', alpha=0.1);
    ax[0,1].plot(f[cal].cumulative_emissions.loc[dict(specie="CO2", scenario="esm-flat10_cdr")], color='r', alpha=0.1, ls=':');
    ax[1,0].plot(np.arange(0, 151), f[cal].concentration.loc[dict(specie="CO2", scenario="esm-flat10", timebounds=np.arange(0, 151))], color='k', alpha=0.1);
    ax[1,0].plot(f[cal].concentration.loc[dict(specie="CO2", scenario="esm-flat10_zec")], color='b', alpha=0.1);
    ax[1,0].plot(f[cal].concentration.loc[dict(specie="CO2", scenario="esm-flat10_cdr")], color='r', alpha=0.1, ls=':');
    ax[1,1].plot(np.arange(0, 151), f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10", timebounds=np.arange(0, 151))], color='k', alpha=0.1);
    ax[1,1].plot(f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10_zec")], color='b', alpha=0.1);
    ax[1,1].plot(f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10_cdr")], color='r', alpha=0.1, ls=':');

# %%
for cal in calibrations:
    fig, ax = pl.subplots()
    pl.plot(np.arange(0, 151), f[cal].airborne_fraction.loc[dict(specie="CO2", scenario="esm-flat10", timebounds=np.arange(0, 151))], color='k', alpha=0.1);
    pl.plot(f[cal].airborne_fraction.loc[dict(specie="CO2", scenario="esm-flat10_zec")], color='b', alpha=0.1);
    pl.plot(np.arange(0, 291), f[cal].airborne_fraction.loc[dict(specie="CO2", scenario="esm-flat10_cdr", timebounds=np.arange(0, 291))], color='r', alpha=0.1, ls=':');

# %%
tcre = {}
zec50 = {}
zec100 = {}
zec200 = {}
tr1000 = {}
tr0 = {}
tpw = {}

for cal in calibrations:
    # TCRE is just warming at year 100
    tcre[cal] = f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10", timebounds=100)]

    # ZEC50 is just warming at year 150 minus year 100
    zec50[cal] = (
        f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10_zec", timebounds=150)] - 
        f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10_zec", timebounds=100)]
    )

    # ZEC100 is just warming at year 200 minus year 100
    zec100[cal] = (
        f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10_zec", timebounds=200)] - 
        f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10_zec", timebounds=100)]
    )

    # ZEC200 is just warming at year 300 minus year 100
    zec200[cal] = (
        f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10_zec", timebounds=300)] - 
        f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10_zec", timebounds=100)]
    )

    # # TNZ can be calculated as a 20 year average around year 150 in esm-flat10-cdr minus a 20 year average around year 125 in esm-flat10
    # tnz = (
    #     f.temperature.loc[dict(layer=0, scenario="esm-flat10_cdr", timebounds=150)] - 
    #     f.temperature.loc[dict(layer=0, scenario="esm-flat10", timebounds=125)]
    # )

    # TR1000 can be calculated as a 20 year average around year 200 in esm-flat10-cdr minus a 20 year average around year 100 in esm-flat10
    tr1000[cal] = (
        f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10_cdr", timebounds=200)] - 
        f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10", timebounds=100)]
    )

    # TR0 can be calculated as a 20 year average around year 310 in esm-flat10-cdr
    tr0[cal] = f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10_cdr", timebounds=310)]

    # Time to Peak Warming (tPW) can be calculated as the time difference between the peak value of 20-year smoothed global mean 
    # temperatures and the point that net zero is achieved in esm-flat10-cdr (year 150)
    tpw[cal] = f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10_cdr")].argmax(axis=0) - 150

# %%
df = {}
for cal in calibrations:
    df[cal] = pd.DataFrame(
        {
            "tcre": tcre[cal],
            "zec50": zec50[cal],
            "zec100": zec100[cal],
            "zec200": zec200[cal],
            "tr1000": tr1000[cal],
            "tr0": tr0[cal],
            "tpw": tpw[cal],
        },
        index = f[cal].configs
    )

# %%
for cal in calibrations:
    sns.pairplot(
        df[cal],
        corner=True,
        plot_kws={"alpha": 0.5},
        height=1,
    )
    pl.suptitle(f'Calibration v{cal}')

# %%
os.makedirs('../output/', exist_ok=True)
for cal in calibrations:
    df[cal].to_csv(f'../output/flat10_key-metrics_fair2.1.3_cal{cal}.csv')

# %%
for cal in calibrations:
    ds = xr.Dataset(
        data_vars=dict(
            temperature=(["time", "scenario", "config"], f[cal].temperature.loc[dict(layer=0)].data),
            co2_concentration=(["time", "scenario", "config"], f[cal].concentration.loc[dict(specie="CO2")].data),
            airborne_fraction=(["time", "scenario", "config"], f[cal].airborne_fraction.loc[dict(specie="CO2")].data),
            ecs=(["config"], f[cal].ebms.ecs.data),
            tcr=(["config"], f[cal].ebms.tcr.data),
            tcre=(["config"], tcre[cal].data),
            zec50=(["config"], zec50[cal].data),
            zec100=(["config"], zec100[cal].data),
            zec200=(["config"], zec200[cal].data),
            tr1000=(["config"], tr1000[cal].data),
            tr0=(["config"], tr0[cal].data),
            tpw=(["config"], tpw[cal].data),
        ),
        coords=dict(
            time=np.arange(321),
            config=list(cal_df[cal].index),
            scenario=scenarios
        ),
    )
    ds.to_netcdf(f'../output/flat10_all-output_fair2.1.3_cal{cal}.nc')

# %%
