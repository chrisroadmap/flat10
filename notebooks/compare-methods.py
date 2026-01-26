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
# # Compare methods
#
# ## First compare methods, then compare calibrations.

# %%
import os
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

# %%
os.makedirs('../plots', exist_ok=True)

# %%
calibrations = ['1.2.0', '1.4.0', '1.4.1']

# %%
df_flat10 = {}
df_1pctCO2 = {}
for cal in calibrations:
    df_flat10[cal] = pd.read_csv(f'../output/flat10_key-metrics_fair2.1.3_cal{cal}.csv', index_col=0)
    df_1pctCO2[cal] = pd.read_csv(f'../output/1pctCO2_key-metrics_fair2.1.3_cal{cal}.csv', index_col=0)

# %%
df_flat10[cal]

# %%
df_1pctCO2[cal]

# %%
low=0.75
high=3
for cal in calibrations:
    fig, ax = pl.subplots(figsize=(12/2.54, 10/2.54))
    sc = ax.scatter(df_flat10[cal]["tcre"], df_1pctCO2[cal]["tcre"], c=df_flat10[cal]["zec50"])
    ax.plot((low, high), (low, high), color='k', lw=3, ls=':')
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("flat10 TCRE (K/TtC)")
    ax.set_ylabel("1pctCO2-1000PgC TCRE (K/Ttc)")
    cb=pl.colorbar(sc)
    cb.set_label('flat10 ZEC50 (K)')
    fig.tight_layout()
    pl.savefig(f'../plots/TCRE_scatter_cal{cal}.png')

# %%
low=-0.4
high=0.9
for cal in calibrations:
    fig, ax = pl.subplots(figsize=(12/2.54, 10/2.54))
    sc = pl.scatter(df_flat10[cal]["zec50"], df_1pctCO2[cal]["zec50"], c=df_flat10[cal]["zec50"])
    ax.plot((low, high), (low, high), color='k', lw=3, ls=':')
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("flat10 ZEC50 (K)")
    ax.set_ylabel("1pctCO2-1000PgC ZEC50 (K)")
    cb=pl.colorbar(sc)
    cb.set_label('flat10 ZEC50 (K)')
    fig.tight_layout()
    pl.savefig(f'../plots/ZEC50_scatter_cal{cal}.png')

# %%
low=-0.6
high=1.3
for cal in calibrations:
    fig, ax = pl.subplots(figsize=(12/2.54, 10/2.54))
    sc = ax.scatter(df_flat10[cal]["zec100"], df_1pctCO2[cal]["zec100"], c=df_flat10[cal]["zec50"])
    ax.plot((low, high), (low, high), color='k', lw=3, ls=':')
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("flat10 ZEC100 (K)")
    ax.set_ylabel("1pctCO2-1000PgC ZEC100 (K)")
    cb=pl.colorbar(sc)
    cb.set_label('flat10 ZEC50 (K)')
    fig.tight_layout()
    pl.savefig(f'../plots/ZEC100_scatter_cal{cal}.png')

# %%
low=0.5
high=3.5
for cal in calibrations:
    fig, ax = pl.subplots(figsize=(12/2.54, 10/2.54))
    sc = ax.scatter(df_flat10[cal]["tcre"] + df_flat10[cal]["zec50"], df_1pctCO2[cal]["tcre"] + df_1pctCO2[cal]["zec50"], c=df_flat10[cal]["zec50"])
    ax.plot((low, high), (low, high), color='k', lw=3, ls=':')
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("flat10 TCRE + ZEC50 (K)")
    ax.set_ylabel("1pctCO2-1000PgC TCRE + ZEC50 (K)")
    cb=pl.colorbar(sc)
    cb.set_label('flat10 ZEC50 (K)')
    fig.tight_layout()
    pl.savefig(f'../plots/TCRE+ZEC50_scatter_cal{cal}.png')

# %%
low=0.7
high=3
dt=0.1
for cal in calibrations:
    fig, ax = pl.subplots(figsize=(10/2.54, 10/2.54))
    ax.hist(df_flat10[cal]["tcre"], histtype='step', lw=2, bins = np.arange(low, high+dt, dt), label='flat10')
    ax.hist(df_1pctCO2[cal]["tcre"], histtype='step', lw=2, bins = np.arange(low, high+dt, dt), label='1pctCO2')
    ax.set_xlabel('TCRE (K/TtC)')
    ax.set_ylabel('Count')
    ax.set_xlim(low, high)
    ax.legend();
    fig.tight_layout()
    pl.savefig(f'../plots/TCRE_hist_cal{cal}.png')

# %%
low=-0.4
high=0.9
dt=0.05
for cal in calibrations:
    fig, ax = pl.subplots(figsize=(10/2.54, 10/2.54))
    ax.hist(df_flat10[cal]["zec50"], histtype='step', lw=2, bins = np.arange(low, high+dt, dt), label='flat10')
    ax.hist(df_1pctCO2[cal]["zec50"], histtype='step', lw=2, bins = np.arange(low, high+dt, dt), label='1pctCO2')
    ax.set_xlabel('ZEC50 (K)')
    ax.set_ylabel('Count')
    ax.set_xlim(low, high)
    ax.legend();
    fig.tight_layout()
    pl.savefig(f'../plots/ZEC50_hist_cal{cal}.png')

# %%
low=-0.6
high=1.3
dt=0.05
for cal in calibrations:
    fig, ax = pl.subplots(figsize=(10/2.54, 10/2.54))
    ax.hist(df_flat10[cal]["zec100"], histtype='step', lw=2, bins = np.arange(low, high+dt, dt), label='flat10')
    ax.hist(df_1pctCO2[cal]["zec100"], histtype='step', lw=2, bins = np.arange(low, high+dt, dt), label='1pctCO2')
    ax.set_xlabel('ZEC100 (K)')
    ax.set_ylabel('Count')
    ax.set_xlim(low, high)
    ax.legend();
    fig.tight_layout()
    pl.savefig(f'../plots/ZEC100_hist_cal{cal}.png')

# %%
low=0.5
high=3.5
dt=0.1
for cal in calibrations:
    fig, ax = pl.subplots(figsize=(10/2.54, 10/2.54))
    ax.hist(df_flat10[cal]["tcre"] + df_flat10[cal]["zec50"], histtype='step', lw=2, bins = np.arange(low, high+dt, dt), label='flat10')
    ax.hist(df_1pctCO2[cal]["tcre"] + df_1pctCO2[cal]["zec50"], histtype='step', lw=2, bins = np.arange(low, high+dt, dt), label='1pctCO2')
    ax.set_xlabel('TCRE + ZEC50 (K)')
    ax.set_ylabel('Count')
    ax.set_xlim(low, high)
    ax.legend();
    fig.tight_layout()
    pl.savefig(f'../plots/TCRE+ZEC50_hist_cal{cal}.png')

# %% [markdown]
# ## Compare calibrations

# %%
low=0.7
high=3
dt=0.1
fig, ax = pl.subplots(figsize=(10/2.54, 10/2.54))
ax.hist(df_1pctCO2['1.4.1']["tcre"], histtype='step', lw=2, bins = np.arange(low, high+dt, dt), label='v1.4.1', color='r')
ax.hist(df_1pctCO2['1.4.0']["tcre"], histtype='step', lw=2, bins = np.arange(low, high+dt, dt), label='v1.4.0', color='k')
ax.set_xlabel('TCRE (K/TtC)')
ax.set_ylabel('Count')
ax.set_xlim(low, high)
ax.axvspan(1.0, 2.3, alpha=0.2, color='k')
ax.legend();
fig.tight_layout()
pl.savefig(f'../plots/TCRE_hist_compare_1.4.0_1.4.1.png')

# %%
low=-0.4
high=0.9
dt=0.05
fig, ax = pl.subplots(figsize=(10/2.54, 10/2.54))
ax.hist(df_1pctCO2['1.4.1']["zec50"], histtype='step', lw=2, bins = np.arange(low, high+dt, dt), label='v1.4.1', color='r')
ax.hist(df_1pctCO2['1.4.0']["zec50"], histtype='step', lw=2, bins = np.arange(low, high+dt, dt), label='v1.4.0', color='k')
ax.set_xlabel('ZEC50 (K)')
ax.set_ylabel('Count')
ax.set_xlim(low, high)
ax.axvline(0, lw=2, color='k', ls=':')
ax.legend();
fig.tight_layout()
pl.savefig(f'../plots/ZEC50_hist_compare_1.4.0_1.4.1.png')

# %%
low=-0.6
high=1.3
dt=0.05
fig, ax = pl.subplots(figsize=(10/2.54, 10/2.54))
ax.hist(df_1pctCO2['1.4.1']["zec100"], histtype='step', lw=2, bins = np.arange(low, high+dt, dt), label='v1.4.1', color='r')
ax.hist(df_1pctCO2['1.4.0']["zec100"], histtype='step', lw=2, bins = np.arange(low, high+dt, dt), label='v1.4.0', color='k')
ax.set_xlabel('ZEC100 (K)')
ax.set_ylabel('Count')
ax.set_xlim(low, high)
ax.axvline(0, lw=2, color='k', ls=':')
ax.legend();
fig.tight_layout()
pl.savefig(f'../plots/ZEC100_hist_compare_1.4.0_1.4.1.png')

# %%
df_1pctCO2['1.4.0']["zec50"].quantile((.05, .50, .95))

# %%
df_1pctCO2['1.4.1']["zec50"].quantile((.05, .50, .95))

# %%
df_1pctCO2['1.4.0']["zec100"].quantile((.05, .50, .95))

# %%
df_1pctCO2['1.4.1']["zec100"].quantile((.05, .50, .95))

# %%
df_1pctCO2['1.4.0']["tcre"].quantile((.05, .50, .95))

# %%
df_1pctCO2['1.4.1']["tcre"].quantile((.05, .50, .95))

# %%
for cal in calibrations:
    fig, ax = pl.subplots(figsize=(12/2.54, 10/2.54))
    sc = pl.scatter(df_1pctCO2[cal]["tcre"], df_1pctCO2[cal]["zec50"], c=df_1pctCO2[cal]["zec50"])
#    ax.plot((low, high), (low, high), color='k', lw=3, ls=':')
#    ax.set_xlim(low, high)
#    ax.set_ylim(low, high)
    ax.set_xlabel("1pctCO2-1000PgC TCRE (K/TtC)")
    ax.set_ylabel("1pctCO2-1000PgC ZEC50 (K)")
    cb=pl.colorbar(sc)
    cb.set_label('1pctCO2-1000PgC ZEC50 (K)')
    fig.tight_layout()
    #pl.savefig(f'../plots/ZEC50_scatter_cal{cal}.png')

# %%
