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
# # FlatX-nGtC runs with FaIR v2.1.3
#
# where X = 1.25, 2.5, 5, 10, 20, 40
#
# and N = 200, 400, 600, 800, 1000, 1200, 1400, 2000, 3000, 5000

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
#scenarios = [f'esm-flat{X}{Z}-{N}GtC' for X in [1.25, 2.5, 5, 10, 20, 40] for Z in ['', '-zec'] for N in [200, 400, 600, 800, 1000, 1200, 1400, 2000, 3000, 5000]]
scenarios = [f'esm-flat{X}-zec-{N}GtC' for X in [1.25, 2.5, 5, 10, 20, 40] for N in [200, 400, 600, 800, 1000, 1200, 1400, 2000, 3000, 5000]]
scenarios

# %%
calibration = '1.4.0'
cal_df = {}

# %%
cal_df['1.4.0'] = pd.read_csv('../data/calibration/v1.4.0/calibrated_constrained_parameters.csv', index_col=0)

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
emissions_rate = np.array([40, 20, 10, 5, 2.5, 1.25])
# ramp_up_length = (1000/emissions_rate).astype(int)
# experiment_length = ramp_up_length + 200
# experiment_length

# %%
emissions_rate

# %%
cumulative_emissions = np.array([200, 400, 600, 800, 1000, 1200, 1400, 2000, 3000, 5000], dtype=int)

# %%
f = {}
for ce in cumulative_emissions:
    f[ce] = {}
    for experiment in emissions_rate:
        if experiment in [40.  , 20.  , 10.  ,  5.]:
            exp_name = int(experiment)
        else:
            exp_name = experiment
        f[ce][exp_name] = FAIR()  # does this explode?
        ramp_up_length = int(ce/experiment)
        experiment_length = ramp_up_length + 5000
        f[ce][exp_name].define_time(0, experiment_length, 1)
#        f[ce][exp_name].define_scenarios([f'esm-flat{exp_name}-{ce}GtC', f'esm-flat{exp_name}-zec-{ce}GtC'])
        f[ce][exp_name].define_scenarios([f'esm-flat{exp_name}-zec-{ce}GtC'])
        f[ce][exp_name].define_configs(list(cal_df[calibration].index))
        
        # declare species and properties
        f[ce][exp_name].define_species(species, properties)
        
        f[ce][exp_name].allocate()
        
        # fill emissions: zero for non-CO2
        f[ce][exp_name].emissions.loc[dict(specie="CH4")] = 0
        f[ce][exp_name].emissions.loc[dict(specie="N2O")] = 0
        
        # constant pre-industrial concentration for non-CO2 GHGs
        f[ce][exp_name].concentration.loc[dict(specie='CH4')] = 808.2490285
        f[ce][exp_name].concentration.loc[dict(specie='N2O')] = 273.021047
        
        # fill emissions of CO2 for each scenario
#        f[ce][exp_name].emissions.loc[dict(specie="CO2", scenario=f'esm-flat{exp_name}-{ce}GtC')] = experiment * 44.009 / 12.011
        f[ce][exp_name].emissions.loc[dict(specie="CO2", scenario=f'esm-flat{exp_name}-zec-{ce}GtC', timepoints=np.arange(0.5, ramp_up_length))] = experiment * 44.009 / 12.011
        f[ce][exp_name].emissions.loc[dict(specie="CO2", scenario=f'esm-flat{exp_name}-zec-{ce}GtC', timepoints=np.arange(ramp_up_length+0.5, experiment_length))] = 0
        
        # Get default species configs
        f[ce][exp_name].fill_species_configs()
    
        # Climate response
        fill(f[ce][exp_name].climate_configs['ocean_heat_capacity'], cal_df[calibration].loc[:,'clim_c1':'clim_c3'])
        fill(f[ce][exp_name].climate_configs['ocean_heat_transfer'], cal_df[calibration].loc[:,'clim_kappa1':'clim_kappa3'])
        fill(f[ce][exp_name].climate_configs['deep_ocean_efficacy'], cal_df[calibration].loc[:,'clim_epsilon'])
        fill(f[ce][exp_name].climate_configs['gamma_autocorrelation'], cal_df[calibration].loc[:,'clim_gamma'])
        fill(f[ce][exp_name].climate_configs['stochastic_run'], False)
    
        # carbon cycle
        fill(f[ce][exp_name].species_configs['iirf_0'], cal_df[calibration].loc[:, 'cc_r0'].values.squeeze(), specie='CO2')
        fill(f[ce][exp_name].species_configs['iirf_airborne'], cal_df[calibration].loc[:, 'cc_rA'].values.squeeze(), specie='CO2')
        fill(f[ce][exp_name].species_configs['iirf_uptake'], cal_df[calibration].loc[:, 'cc_rU'].values.squeeze(), specie='CO2')
        fill(f[ce][exp_name].species_configs['iirf_temperature'], cal_df[calibration].loc[:, 'cc_rT'].values.squeeze(), specie='CO2')
    
        # Scale CO2 forcing based on its 4xCO2 calibration
        fill(f[ce][exp_name].species_configs["forcing_scale"], cal_df[calibration]["fscale_CO2"].values.squeeze(), specie='CO2')
    
        # initial condition of CO2 concentration (but not baseline for forcing calculations)
        fill(f[ce][exp_name].species_configs['baseline_concentration'], 284.3169988, specie='CO2')
        fill(f[ce][exp_name].species_configs['baseline_concentration'], 808.2490285, specie='CH4')
        fill(f[ce][exp_name].species_configs['baseline_concentration'], 273.021047, specie='N2O')
        
        # set initial conditions
        initialise(f[ce][exp_name].concentration, f[ce][exp_name].species_configs['baseline_concentration'])
        initialise(f[ce][exp_name].forcing, 0)
        initialise(f[ce][exp_name].temperature, 0)
        initialise(f[ce][exp_name].airborne_emissions, 0)
        initialise(f[ce][exp_name].cumulative_emissions, 0)
        
        f[ce][exp_name].run()

# %%
# for experiment in emissions_rate:
#     if experiment in [40.  , 20.  , 10.  ,  5.]:
#         exp_name = int(experiment)
#     fig, ax = pl.subplots(2, 2)
#     ax[0,0].plot(f[exp_name].timepoints, f[exp_name].emissions.loc[dict(specie="CO2", scenario=f"esm-flat{exp_name}-1000GtC")], color='k', alpha=0.1);
#     ax[0,0].plot(f[exp_name].timepoints, f[exp_name].emissions.loc[dict(specie="CO2", scenario=f"esm-flat{exp_name}-zec-1000GtC")], color='b', alpha=0.1);
#     ax[0,1].plot(f[exp_name].timebounds, f[exp_name].cumulative_emissions.loc[dict(specie="CO2", scenario=f"esm-flat{exp_name}-1000GtC")], color='k', alpha=0.1);
#     ax[0,1].plot(f[exp_name].cumulative_emissions.loc[dict(specie="CO2", scenario=f"esm-flat{exp_name}-zec-1000GtC")], color='b', alpha=0.1);
#     ax[1,0].plot(f[exp_name].timebounds, f[exp_name].concentration.loc[dict(specie="CO2", scenario=f"esm-flat{exp_name}-1000GtC")], color='k', alpha=0.1);
#     ax[1,0].plot(f[exp_name].concentration.loc[dict(specie="CO2", scenario=f"esm-flat{exp_name}-zec-1000GtC")], color='b', alpha=0.1);
#     ax[1,1].plot(f[exp_name].timebounds, f[exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-1000GtC")], color='k', alpha=0.1);
#     ax[1,1].plot(f[exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-1000GtC")], color='b', alpha=0.1);

# %%
# for experiment in emissions_rate:
#     if experiment in [40.  , 20.  , 10.  ,  5.]:
#         exp_name = int(experiment)
#     fig, ax = pl.subplots()
#     pl.plot(f[exp_name].timebounds, f[exp_name].airborne_fraction.loc[dict(specie="CO2", scenario=f"esm-flat{exp_name}-1000GtC")], color='k', alpha=0.1);
#     pl.plot(f[exp_name].timebounds, f[exp_name].airborne_fraction.loc[dict(specie="CO2", scenario=f"esm-flat{exp_name}-zec-1000GtC")], color='b', alpha=0.1);

# %%
tcre = {}
zec50 = {}
zec100 = {}
zec200 = {}
zec1000 = {}
zec5000 = {}
total50 = {}
total100 = {}
total200 = {}
total1000 = {}
total5000 = {}
totalpeak = {}
# tr1000 = {}
# tr0 = {}
tpw = {}

for ce in cumulative_emissions:
    tcre[ce] = {}
    zec50[ce] = {}
    zec100[ce] = {}
    zec200[ce] = {}
    zec1000[ce] = {}
    zec5000[ce] = {}
    total50[ce] = {}
    total100[ce] = {}
    total200[ce] = {}
    total1000[ce] = {}
    total5000[ce] = {}
    totalpeak[ce] = {}
    tpw[ce] = {}
    
    for experiment in emissions_rate:
        if experiment in [40.  , 20.  , 10.  ,  5.]:
            exp_name = int(experiment)
        else:
            exp_name = experiment
        ramp_up_length = int(ce/experiment)
        # TCRE; normalised
        tcre[ce][exp_name] = f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length)]/ce * 1000
    
        # ZEC50
        zec50[ce][exp_name] = (
            f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length+50)] - 
            f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length)]
        )
    
        # ZEC100
        zec100[ce][exp_name] = (
            f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length+100)] - 
            f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length)]
        )
    
        # ZEC200
        zec200[ce][exp_name] = (
            f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length+200)] - 
            f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length)]
        )

        # ZEC1000
        zec1000[ce][exp_name] = (
            f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length+1000)] - 
            f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length)]
        )

        # ZEC5000
        zec5000[ce][exp_name] = (
            f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length+5000)] - 
            f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length)]
        )
    
        # total50 is the total warming 50 years after emissions reach zero
        total50[ce][exp_name] = (
            f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length+50)]
        )

        # total100 is the total warming 500 years after emissions reach zero
        total100[ce][exp_name] = (
            f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length+100)]
        )

        # total200 is the total warming 200 years after emissions reach zero
        total200[ce][exp_name] = (
            f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length+200)]
        )

        # total1000 is the total warming 1000 years after emissions reach zero
        total1000[ce][exp_name] = (
            f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length+1000)]
        )

        # total5000 is the total warming 5000 years after emissions reach zero
        total5000[ce][exp_name] = (
            f[ce][exp_name].temperature.loc[dict(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC", timebounds=ramp_up_length+5000)]
        )

        totalpeak[ce][exp_name] = (
            f[ce][exp_name].temperature.sel(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC").max(dim='timebounds')
        )
        
        # # TNZ can be calculated as a 20 year average around year 150 in esm-flat10-cdr minus a 20 year average around year 125 in esm-flat10
        # tnz = (
        #     f.temperature.loc[dict(layer=0, scenario="esm-flat10_cdr", timebounds=150)] - 
        #     f.temperature.loc[dict(layer=0, scenario="esm-flat10", timebounds=125)]
        # )
    
        # # TR1000 can be calculated as a 20 year average around year 200 in esm-flat10-cdr minus a 20 year average around year 100 in esm-flat10
        # tr1000[cal] = (
        #     f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10_cdr", timebounds=200)] - 
        #     f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10", timebounds=100)]
        # )
    
        # # TR0 can be calculated as a 20 year average around year 310 in esm-flat10-cdr
        # tr0[cal] = f[cal].temperature.loc[dict(layer=0, scenario="esm-flat10_cdr", timebounds=310)]
    
        # Time to Peak Warming (tPW) can be calculated as the time difference between the peak value of 20-year smoothed global mean 
        # temperatures and the point that net zero is achieved in esm-flat10-cdr (year 150)
        tpw[ce][exp_name] = f[ce][exp_name].temperature.sel(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC").argmax(axis=0) - ramp_up_length

# %%
tpw

# %%
#pl.plot(f[5000][40].timebounds, f[5000][40].temperature.loc[dict(layer=0, scenario="esm-flat40-5000GtC")], color='k', alpha=0.1);
pl.plot(f[5000][40].temperature.loc[dict(layer=0, scenario="esm-flat40-zec-5000GtC")], color='b', alpha=0.1);

# %%
pl.plot(f[5000][1.25].temperature.loc[dict(layer=0, scenario="esm-flat1.25-zec-5000GtC")], color='b', alpha=0.1);

# %%
df = {}
for ce in cumulative_emissions:
    df[ce] = {}
    for experiment in emissions_rate:
        if experiment in [40.  , 20.  , 10.  ,  5.]:
            exp_name = int(experiment)
        else:
            exp_name = experiment
        df[ce][exp_name] = pd.DataFrame(
            {
                "tcre": tcre[ce][exp_name],
                "zec50": zec50[ce][exp_name],
                "zec100": zec100[ce][exp_name],
                "zec200": zec200[ce][exp_name],
                "zec1000": zec1000[ce][exp_name],
                "zec5000": zec5000[ce][exp_name],
                "total50": total50[ce][exp_name],
                "total100": total100[ce][exp_name],
                "total200": total200[ce][exp_name],
                "total1000": total1000[ce][exp_name],
                "total5000": total5000[ce][exp_name],
                "totalpeak": totalpeak[ce][exp_name],
                "tpw": tpw[ce][exp_name],
            },
            index = f[ce][exp_name].configs
        )

# %%
df[5000][40]

# %%
df[5000][40]["total50"].quantile((.05, .50, .95))

# %%
df[5000][20]["total50"].quantile((.05, .50, .95))

# %%
df[5000][10]["total50"].quantile((.05, .50, .95))

# %%
df[5000][5]["total50"].quantile((.05, .50, .95))

# %%
df[5000][2.5]["total50"].quantile((.05, .50, .95))

# %%
df[5000][1.25]["total50"].quantile((.05, .50, .95))

# %%
df[5000][40]["zec50"].quantile((.05, .50, .95))

# %%
df[5000][20]["zec50"].quantile((.05, .50, .95))

# %%
df[5000][10]["zec50"].quantile((.05, .50, .95))

# %%
df[5000][5]["zec50"].quantile((.05, .50, .95))

# %%
df[5000][2.5]["zec50"].quantile((.05, .50, .95))

# %%
df[5000][1.25]["zec50"].quantile((.05, .50, .95))

# %%
df[5000][40]["zec100"].quantile((.05, .50, .95))

# %%
df[5000][20]["zec100"].quantile((.05, .50, .95))

# %%
df[5000][10]["zec100"].quantile((.05, .50, .95))

# %%
df[5000][5]["zec100"].quantile((.05, .50, .95))

# %%
df[5000][2.5]["zec100"].quantile((.05, .50, .95))

# %%
df[5000][1.25]["zec100"].quantile((.05, .50, .95))

# %%
df[5000][40]["tcre"].quantile((.05, .50, .95))

# %%
df[5000][20]["tcre"].quantile((.05, .50, .95))

# %%
df[5000][10]["tcre"].quantile((.05, .50, .95))

# %%
df[5000][5]["tcre"].quantile((.05, .50, .95))

# %%
df[5000][2.5]["tcre"].quantile((.05, .50, .95))

# %%
df[5000][1.25]["tcre"].quantile((.05, .50, .95))

# %%
matrix_50_zec50 = np.zeros((6, 10))
for ien, expt_name in enumerate([1.25, 2.5, 5, 10, 20, 40]):
    for ice, ce in enumerate(cumulative_emissions):
        matrix_50_zec50[ien, ice] = df[ce][expt_name]["zec50"].quantile(.50)

# %%
x, y = np.meshgrid(np.array([1.25, 2.5, 5, 10, 20, 40]), cumulative_emissions)

# %%
pl.contourf(x, y, matrix_50_zec50.T)
pl.colorbar()
pl.xlabel('Emissions rate, GtC/yr')
pl.ylabel('Cumulative emissions, GtC')
pl.title('ZEC50')

# %%
matrix_50_zec100 = np.zeros((6, 10))
for ien, expt_name in enumerate([1.25, 2.5, 5, 10, 20, 40]):
    for ice, ce in enumerate(cumulative_emissions):
        matrix_50_zec100[ien, ice] = df[ce][expt_name]["zec100"].quantile(.50)

# %%
pl.contourf(x, y, matrix_50_zec100.T)
pl.colorbar()
pl.xlabel('Emissions rate, GtC/yr')
pl.ylabel('Cumulative emissions, GtC')
pl.title('ZEC100')

# %%
matrix_50_zec200 = np.zeros((6, 10))
for ien, expt_name in enumerate([1.25, 2.5, 5, 10, 20, 40]):
    for ice, ce in enumerate(cumulative_emissions):
        matrix_50_zec200[ien, ice] = df[ce][expt_name]["zec200"].quantile(.50)

# %%
pl.contourf(x, y, matrix_50_zec200.T)
pl.colorbar()
pl.xlabel('Emissions rate, GtC/yr')
pl.ylabel('Cumulative emissions, GtC')
pl.title('ZEC200')

# %%
matrix_50_tcre = np.zeros((6, 10))
for ien, expt_name in enumerate([1.25, 2.5, 5, 10, 20, 40]):
    for ice, ce in enumerate(cumulative_emissions):
        matrix_50_tcre[ien, ice] = df[ce][expt_name]["tcre"].quantile(.50)

# %%
pl.contourf(x, y, matrix_50_tcre.T)
pl.colorbar()
pl.xlabel('Emissions rate, GtC/yr')
pl.ylabel('Cumulative emissions, GtC')
pl.title('TCRE')

# %%
matrix_50_total50 = np.zeros((6, 10))
for ien, expt_name in enumerate([1.25, 2.5, 5, 10, 20, 40]):
    for ice, ce in enumerate(cumulative_emissions):
        matrix_50_total50[ien, ice] = df[ce][expt_name]["total50"].quantile(.50)/ce * 1000

# %%
pl.contourf(x, y, matrix_50_total50.T)
pl.colorbar()
pl.xlabel('Emissions rate, GtC/yr')
pl.ylabel('Cumulative emissions, GtC')
pl.title('(TCRE + ZEC50) ÷ (cum. emissions)')

# %%
matrix_50_total200 = np.zeros((6, 10))
for ien, expt_name in enumerate([1.25, 2.5, 5, 10, 20, 40]):
    for ice, ce in enumerate(cumulative_emissions):
        matrix_50_total200[ien, ice] = df[ce][expt_name]["total200"].quantile(.50)/ce * 1000

# %%
pl.contourf(x, y, matrix_50_total200.T)
pl.colorbar()
pl.xlabel('Emissions rate, GtC/yr')
pl.ylabel('Cumulative emissions, GtC')
pl.title('(TCRE + ZEC200) ÷ (cum. emissions)')

# %%
matrix_50_total1000 = np.zeros((6, 10))
for ien, expt_name in enumerate([1.25, 2.5, 5, 10, 20, 40]):
    for ice, ce in enumerate(cumulative_emissions):
        matrix_50_total1000[ien, ice] = df[ce][expt_name]["total1000"].quantile(.50)/ce * 1000

# %%
pl.contourf(x, y, matrix_50_total1000.T)
pl.colorbar()
pl.xlabel('Emissions rate, GtC/yr')
pl.ylabel('Cumulative emissions, GtC')
pl.title('(TCRE + ZEC1000) ÷ (cum. emissions)')

# %%
matrix_50_totalpeak = np.zeros((6, 10))
for ien, expt_name in enumerate([1.25, 2.5, 5, 10, 20, 40]):
    for ice, ce in enumerate(cumulative_emissions):
        matrix_50_totalpeak[ien, ice] = df[ce][expt_name]["totalpeak"].quantile(.50)/ce * 1000

# %%
pl.contourf(x, y, matrix_50_totalpeak.T)
pl.colorbar()
pl.xlabel('Emissions rate, GtC/yr')
pl.ylabel('Cumulative emissions, GtC')
pl.title('(peak warming) ÷ (cum. emissions)')

# %%
pl.plot(
    np.array([1.25, 2.5, 5, 10, 20, 40]),
    [
        df[1000][1.25]["tcre"].quantile((.05)), 
        df[1000][2.5]["tcre"].quantile((.05)), 
        df[1000][5]["tcre"].quantile((.05)), 
        df[1000][10]["tcre"].quantile((.05)), 
        df[1000][20]["tcre"].quantile((.05)), 
        df[1000][40]["tcre"].quantile((.05))
    ],
    ls = '--',
    marker='o',
    color='r',
),
pl.plot(
    np.array([1.25, 2.5, 5, 10, 20, 40]),
    [
        df[1000][1.25]["tcre"].quantile((.95)), 
        df[1000][2.5]["tcre"].quantile((.95)), 
        df[1000][5]["tcre"].quantile((.95)), 
        df[1000][10]["tcre"].quantile((.95)), 
        df[1000][20]["tcre"].quantile((.95)), 
        df[1000][40]["tcre"].quantile((.95))
    ],
    ls = '--',
    marker='o',
    color='r',
)
pl.plot(
    np.array([1.25, 2.5, 5, 10, 20, 40]),
    [
        df[1000][1.25]["tcre"].quantile((.5)), 
        df[1000][2.5]["tcre"].quantile((.5)), 
        df[1000][5]["tcre"].quantile((.5)), 
        df[1000][10]["tcre"].quantile((.5)), 
        df[1000][20]["tcre"].quantile((.5)), 
        df[1000][40]["tcre"].quantile((.5))
    ],
    ls = '-',
    marker='o',
    color='k',
)
pl.xlabel('Emissions rate, GtC/yr')
pl.ylabel('TCRE, K at 1000 GtC')
pl.savefig('../plots/TCRE_flatX.png')

# %%
pl.plot(
    np.array([1.25, 2.5, 5, 10, 20, 40]),
    [
        df[1000][1.25]["total50"].quantile((.05)), 
        df[1000][2.5]["total50"].quantile((.05)), 
        df[1000][5]["total50"].quantile((.05)), 
        df[1000][10]["total50"].quantile((.05)), 
        df[1000][20]["total50"].quantile((.05)), 
        df[1000][40]["total50"].quantile((.05))
    ],
    ls = '--',
    marker='o',
    color='r',
),
pl.plot(
    np.array([1.25, 2.5, 5, 10, 20, 40]),
    [
        df[1000][1.25]["total50"].quantile((.95)), 
        df[1000][2.5]["total50"].quantile((.95)), 
        df[1000][5]["total50"].quantile((.95)), 
        df[1000][10]["total50"].quantile((.95)), 
        df[1000][20]["total50"].quantile((.95)), 
        df[1000][40]["total50"].quantile((.95))
    ],
    ls = '--',
    marker='o',
    color='r',
)
pl.plot(
    np.array([1.25, 2.5, 5, 10, 20, 40]),
    [
        df[1000][1.25]["total50"].quantile((.5)), 
        df[1000][2.5]["total50"].quantile((.5)), 
        df[1000][5]["total50"].quantile((.5)), 
        df[1000][10]["total50"].quantile((.5)), 
        df[1000][20]["total50"].quantile((.5)), 
        df[1000][40]["total50"].quantile((.5))
    ],
    ls = '-',
    marker='o',
    color='k',
)
pl.xlabel('Emissions rate, GtC/yr')
pl.ylabel('TCRE + ZEC50, K at 1000 GtC')
pl.savefig('../plots/TCRE+ZEC50_flatX.png')

# %%
pl.plot(
    np.array([1.25, 2.5, 5, 10, 20, 40]),
    [
        df[1000][1.25]["zec50"].quantile((.05)), 
        df[1000][2.5]["zec50"].quantile((.05)), 
        df[1000][5]["zec50"].quantile((.05)), 
        df[1000][10]["zec50"].quantile((.05)), 
        df[1000][20]["zec50"].quantile((.05)), 
        df[1000][40]["zec50"].quantile((.05))
    ],
    ls = '--',
    marker='o',
    color='r',
),
pl.plot(
    np.array([1.25, 2.5, 5, 10, 20, 40]),
    [
        df[1000][1.25]["zec50"].quantile((.95)), 
        df[1000][2.5]["zec50"].quantile((.95)), 
        df[1000][5]["zec50"].quantile((.95)), 
        df[1000][10]["zec50"].quantile((.95)), 
        df[1000][20]["zec50"].quantile((.95)), 
        df[1000][40]["zec50"].quantile((.95))
    ],
    ls = '--',
    marker='o',
    color='r',
)
pl.plot(
    np.array([1.25, 2.5, 5, 10, 20, 40]),
    [
        df[1000][1.25]["zec50"].quantile((.5)), 
        df[1000][2.5]["zec50"].quantile((.5)), 
        df[1000][5]["zec50"].quantile((.5)), 
        df[1000][10]["zec50"].quantile((.5)), 
        df[1000][20]["zec50"].quantile((.5)), 
        df[1000][40]["zec50"].quantile((.5))
    ],
    ls = '-',
    marker='o',
    color='k',
)
pl.xlabel('Emissions rate, GtC/yr')
pl.ylabel('ZEC50, K after 1000 GtC')
pl.savefig('../plots/ZEC50_flatX.png')

# %%
pl.plot(
    np.array([1.25, 2.5, 5, 10, 20, 40]),
    [
        df[1000][1.25]["zec100"].quantile((.05)), 
        df[1000][2.5]["zec100"].quantile((.05)), 
        df[1000][5]["zec100"].quantile((.05)), 
        df[1000][10]["zec100"].quantile((.05)), 
        df[1000][20]["zec100"].quantile((.05)), 
        df[1000][40]["zec100"].quantile((.05))
    ],
    ls = '--',
    marker='o',
    color='r',
),
pl.plot(
    np.array([1.25, 2.5, 5, 10, 20, 40]),
    [
        df[1000][1.25]["zec100"].quantile((.95)), 
        df[1000][2.5]["zec100"].quantile((.95)), 
        df[1000][5]["zec100"].quantile((.95)), 
        df[1000][10]["zec100"].quantile((.95)), 
        df[1000][20]["zec100"].quantile((.95)), 
        df[1000][40]["zec100"].quantile((.95))
    ],
    ls = '--',
    marker='o',
    color='r',
)
pl.plot(
    np.array([1.25, 2.5, 5, 10, 20, 40]),
    [
        df[1000][1.25]["zec100"].quantile((.5)), 
        df[1000][2.5]["zec100"].quantile((.5)), 
        df[1000][5]["zec100"].quantile((.5)), 
        df[1000][10]["zec100"].quantile((.5)), 
        df[1000][20]["zec100"].quantile((.5)), 
        df[1000][40]["zec100"].quantile((.5))
    ],
    ls = '-',
    marker='o',
    color='k',
)
pl.xlabel('Emissions rate, GtC/yr')
pl.ylabel('zec100, K after 1000 GtC')
pl.savefig('../plots/ZEC100_flatX.png')

# %%
pl.plot(
    np.array([1.25, 2.5, 5, 10, 20, 40]),
    [
        df[1000][1.25]["zec200"].quantile((.05)),
        df[1000][2.5]["zec200"].quantile((.05)), 
        df[1000][5]["zec200"].quantile((.05)), 
        df[1000][10]["zec200"].quantile((.05)), 
        df[1000][20]["zec200"].quantile((.05)), 
        df[1000][40]["zec200"].quantile((.05))
    ],
    ls = '--',
    marker='o',
    color='r',
),
pl.plot(
    np.array([1.25, 2.5, 5, 10, 20, 40]),
    [
        df[1000][1.25]["zec200"].quantile((.95)), 
        df[1000][2.5]["zec200"].quantile((.95)), 
        df[1000][5]["zec200"].quantile((.95)), 
        df[1000][10]["zec200"].quantile((.95)), 
        df[1000][20]["zec200"].quantile((.95)), 
        df[1000][40]["zec200"].quantile((.95))
    ],
    ls = '--',
    marker='o',
    color='r',
)
pl.plot(
    np.array([1.25, 2.5, 5, 10, 20, 40]),
    [
        df[1000][1.25]["zec200"].quantile((.5)), 
        df[1000][2.5]["zec200"].quantile((.5)), 
        df[1000][5]["zec200"].quantile((.5)), 
        df[1000][10]["zec200"].quantile((.5)), 
        df[1000][20]["zec200"].quantile((.5)), 
        df[1000][40]["zec200"].quantile((.5))
    ],
    ls = '-',
    marker='o',
    color='k',
)
pl.xlabel('Emissions rate, GtC/yr')
pl.ylabel('zec200, K after 1000 GtC')
pl.savefig('../plots/ZEC200_flatX.png')

# %%
for experiment in emissions_rate:
    if experiment in [40.  , 20.  , 10.  ,  5.]:
        exp_name = int(experiment)
    else:
        exp_name = experiment
    sns.pairplot(
        df[1000][exp_name],
        corner=True,
        plot_kws={"alpha": 0.5},
        height=1,
    )
    pl.suptitle(f'{experiment} GtC/yr')

# %%
os.makedirs('../output/flatX', exist_ok=True)
for ice, ce in enumerate(cumulative_emissions):
    for experiment in emissions_rate:
        if experiment in [40.  , 20.  , 10.  ,  5.]:
            exp_name = int(experiment)
        else:
            exp_name = experiment
        df[ce][exp_name].to_csv(f'../output/flatX/esm-flat{exp_name}-zec-{ce}GtC_key-metrics_fair2.1.3_cal1.4.0.csv')

# %%
os.makedirs('../output/flatX', exist_ok=True)
for ice, ce in enumerate(cumulative_emissions):
    for experiment in emissions_rate:
        if experiment in [40.  , 20.  , 10.  ,  5.]:
            exp_name = int(experiment)
        else:
            exp_name = experiment
        f[ce][exp_name].temperature.sel(layer=0, scenario=f"esm-flat{exp_name}-zec-{ce}GtC").to_netcdf(f'../output/flatX/esm-flat{exp_name}-zec-{ce}GtC_temperature_fair2.1.3_cal1.4.0.nc')

# %%
# for cal in calibrations:
#     ds = xr.Dataset(
#         data_vars=dict(
#             temperature=(["time", "scenario", "config"], f[cal].temperature.loc[dict(layer=0)].data),
#             co2_concentration=(["time", "scenario", "config"], f[cal].concentration.loc[dict(specie="CO2")].data),
#             airborne_fraction=(["time", "scenario", "config"], f[cal].airborne_fraction.loc[dict(specie="CO2")].data),
#             ecs=(["config"], f[cal].ebms.ecs.data),
#             tcr=(["config"], f[cal].ebms.tcr.data),
#             tcre=(["config"], tcre[cal].data),
#             zec50=(["config"], zec50[cal].data),
#             zec100=(["config"], zec100[cal].data),
#             zec200=(["config"], zec200[cal].data),
#             # tr1000=(["config"], tr1000[cal].data),
#             # tr0=(["config"], tr0[cal].data),
#             # tpw=(["config"], tpw[cal].data),
#         ),
#         coords=dict(
#             time=np.arange(226),
#             config=list(cal_df[cal].index),
#             scenario=scenarios
#         ),
#     )
#     ds.to_netcdf(f'../output/flat40_all-output_fair2.1.3_cal{cal}.nc')

# %%
