import sys
import os
sys.path.append('/g/data/jk72/ll6859/access_aero_eval/')
from aercode import *
import pandas as pd
import numpy as np
import xarray as xr
import glob as gb
import scipy.special
import datetime as dt

print("START")

# set path to data
in_path = '/g/data/jk72/ll6859/access_aero_eval/data/'
print("INPUT PATH SET AS: {}".format(in_path))

# set path to output
out_path = '/g/data/jk72/ll6859/access_aero_eval/output/'
print("OUTPUT PATH SET AS: {}".format(out_path))

# Select and print model variables
variables = ['lat','lon','N3','N10','CCN40','CCN50','CCN60','uas','vas','psl','pr','tempk'] # add or remove variables to keep here
print("Selected variables: ",variables)

# Select and print model runs
runs = ['bx400', 'cg283'] # add or remove model runs here
print("Selected run: ",runs)

# define pipeline
def pipeline(key, run):
    # set file path extensions
    m_path = '/g/data/jk72/slf563/ACCESS/output/{}/daily/'.format(run)
    mod = '{}a'.format(run)
    aer_dt = 'pd.sh'
    met_dt = 'pd.glob'
    key = uw

    print("STARTING: {} {}".format(v_name, run))
    
    # extract aerosol data
    aertrack = df_md(key, m_path, mod, aer_dt)
    
    # extract meteorology data
    mettrack = df_md(key, m_path, mod, met_dt)
    
    # calculate air density from temperature
    mettrack = calc_density(mettrack)
    
    # convert aerosol units
    aertrack = aero_unit_conversions(aertrack, mettrack)
    
    # select data at 20m
    aertrack = aertrack.isel(z1_hybrid_height=0).expand_dims('z1_hybrid_height').transpose()
    mettrack = mettrack.isel(z1_hybrid_height=0, z0_hybrid_height=0).expand_dims('z1_hybrid_height').transpose()
    mettrack = mettrack.drop('z0_hybrid_height')
    
    # calculate aerosol number concentrations
    aertrack = nt_calcs(aertrack)
    
    # calculate CCN number concentrations
    aertrack = ccn_calcs(aertrack)

    # merge the aerosol and meteorology data together
    track = xr.merge([aertrack,mettrack])
    
    # select only required variables
    track = track[variables]
    
    # remove time component from datetime coordinate
    track['time'] = track.indexes['time'].normalize()

    # load and save file
    name = '{}_{}_track.nc'.format(v_name, run)
    track.load().to_netcdf(path=out_path+name)
    print("COMPLETED: {} {}".format(v_name, run))

print("PIPELINE LOADED")

# Start tracks
# RVI
print("TRACKING: RVInvestigator")
v_name = 'rvi1619'
uw = keycutter(in_path+'rvigaw_cn10_2016to2019_L2.csv', 'datetime', 'latitude', 'longitude')
uw = uw.dropna()

for run in runs:
    pipeline(uw, run)

# CWT
print("TRACKING: Cold Water Trial")
v_name = 'cwt15'
uw = keycutter(in_path+'CWT_ProcessedAerosolData_1min.csv', 'date', 'lat', 'lon')
uw = uw.dropna()

for run in runs:
    pipeline(uw, run)

# R2R
print("TRACKING: Reef 2 Rainforest")
v_name = 'r2r16'
uw = keycutter(in_path+'R2R_RV_aerosol.csv', 'date', 'lat', 'lon')
uw = uw.dropna()

for run in runs:
    pipeline(uw, run)

# I2E
print("TRACKING: Ice 2 Equator")
v_name = 'i2e16'
uw = keycutter(in_path+'I2E_ProcessedAerosolData_1min.csv', 'date', 'lat', 'lon')
uw = uw.dropna()

for run in runs:
    pipeline(uw, run)

# Capricorn 1
print("TRACKING: CAPRICORN 1")
v_name = 'cap16'
uw = pd.read_csv(in_path+'in2016_v02uwy10sec.csv', parse_dates=[['date', 'time']], index_col='date_time')
uw = uw.resample('1D').mean()
uw = uw[['latitude(degree_north)', 'longitude(degree_east)']]
uw.columns = ['lat','lon']

for run in runs:
    pipeline(uw, run)

# Capricorn 2
print("TRACKING: CAPRICORN 2")
v_name = 'cap18'
uw = pd.read_csv(in_path+'in2018_v01uwy10sec.csv', parse_dates=[['date', 'time']], index_col='date_time')
for v in uw.select_dtypes('object'):
    uw[v] = pd.to_numeric(uw[v], errors='coerce')
uw.index.name = 'time'
uw = uw.resample('1D').mean()
uw = uw[['latitude(degree_north)','longitude(degree_east)']]
uw.rename(columns = {"latitude(degree_north)": "lat",
                            "longitude(degree_east)": "lon"}, inplace=True)

for run in runs:
    pipeline(uw, run)

# MARCUS
print("TRACKING: MARCUS")
v_name = 'aa1718'
uw = keycutter(in_path+'marcus_uw/I2E_ProcessedAerosolData_1min.csv', 'date', 'lat', 'lon')
uw = uw.dropna()

for run in runs:
    pipeline(uw, run)

# CAMMPCAN
print("TRACKING: CAMMPCAN")
v_name = 'aa1819'
voyages = sorted(gb.glob(in_path+'cammpcan_uw/201819_Voyage*.csv'))
uw = pd.concat([pd.read_csv(v) for v in voyages ])
uw = uw.set_index(pd.to_datetime(uw['date_time_utc']))
uw = uw[['latitude','longitude']]
uw = uw.resample('1D', kind='Date').mean().ffill()
uw.columns = ['lat','lon']

for run in runs:
    pipeline(uw, run)

# TAN1802
print("TRACKING: TAN1802")
v_name = 'tan1718'
uw = keycutter(in_path+'weather.csv', 'Time (UTC)', 'Latitude', 'Longitude')
uw = uw.dropna()

# MQI
print("TRACKING: MACQUARIE ISLAND")
v_name = 'mqi1618'
uw = pd.read_hdf(in_path+'ACRE_CN10_1H_FINAL.h5')
uw = uw.resample('1D', kind='date_time').mean()
uw['lat'] = -54.38
uw['lon'] = 158.4
uw = uw[['lat','lon']]

for run in runs:
    pipeline(uw, run)

# CGO
print("TRACKING: CAPE GRIM")
v_name = 'cgo1618'
uw = pd.read_excel(in_path+'cgbpaps_CN  2016-2018.xlsx', index_col='date')
uw = uw.resample('1D', kind='date_time').mean()
uw['lat'] = -40.68333
uw['lon'] = 144.6833
uw = uw[['lat','lon']]

print("DONE")