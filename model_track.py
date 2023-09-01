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

# set which campaigns to track (TRUE or FALSE)
run_rvi = False
run_cwt = False
run_r2r = False
run_i2e = False
run_cap1 = False
run_cap2 = False
run_marcus = False
run_cammpcan = False
run_tan1802 = True
run_mqi = False
run_cgo = False

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
    print("Extracting aerosol data...")
    aertrack = df_md(key, m_path, mod, aer_dt)
    
    # extract meteorology data
    print("Extracting meteorology data...")
    mettrack = df_md(key, m_path, mod, met_dt)
    
    # calculate air density from temperature
    print("Calculating density...")
    mettrack = calc_density(mettrack)
    
    # convert aerosol units
    print("Converting units...")
    aertrack = aero_unit_conversions(aertrack, mettrack)
    
    # select data at 20m
    print("Selecting surface level data...")
    aertrack = aertrack.isel(z1_hybrid_height=0).expand_dims('z1_hybrid_height').transpose()
    mettrack = mettrack.isel(z1_hybrid_height=0, z0_hybrid_height=0).expand_dims('z1_hybrid_height').transpose()
    mettrack = mettrack.drop('z0_hybrid_height')
    
    # calculate aerosol number concentrations
    print("Calculating aerosol number concentrations...")
    aertrack = nt_calcs(aertrack)
    
    # calculate CCN number concentrations
    print("Calculating CCN number concentrations...")
    aertrack = ccn_calcs(aertrack)

    # merge the aerosol and meteorology data together
    print("Merging data...")
    track = xr.merge([aertrack,mettrack])
    
    # select only required variables
    print("Selecting variables...")
    track = track[variables]
    
    # remove time component from datetime coordinate
    print("Normalizing time...")
    track['time'] = track.indexes['time'].normalize()

    # load and save file
    print("Saving data as netCDF...")
    name = '{}_{}_track.nc'.format(v_name, run)
    track.load().to_netcdf(path=out_path+name)
    print("COMPLETED: {} {}".format(v_name, run))

print("PIPELINE LOADED")

# Start tracks
# RVI
if run_rvi == True:
    print("TRACKING: RVInvestigator")
    v_name = 'rvi1619'
    uw = keycutter(in_path+'rvigaw_cn10_2016to2019_L2.csv', 'datetime', 'latitude', 'longitude')
    uw = uw.dropna()
    
    for run in runs:
        pipeline(uw, run)

# CWT
if run_cwt == True:
    print("TRACKING: Cold Water Trial")
    v_name = 'cwt15'
    uw = keycutter(in_path+'CWT_ProcessedAerosolData_1min.csv', 'date', 'lat', 'lon')
    uw = uw.dropna()
    
    for run in runs:
        pipeline(uw, run)

# R2R
if run_r2r == True:
    print("TRACKING: Reef 2 Rainforest")
    v_name = 'r2r16'
    uw = keycutter(in_path+'R2R_RV_aerosol.csv', 'date', 'lat', 'lon')
    uw = uw.dropna()
    
    for run in runs:
        pipeline(uw, run)

# I2E
if run_i2e == True:
    print("TRACKING: Ice 2 Equator")
    v_name = 'i2e16'
    uw = keycutter(in_path+'I2E_ProcessedAerosolData_1min.csv', 'date', 'lat', 'lon')
    uw = uw.dropna()
    
    for run in runs:
        pipeline(uw, run)

# Capricorn 1
if run_cap1 == True:
    print("TRACKING: CAPRICORN 1")
    v_name = 'cap16'
    uw = pd.read_csv(in_path+'in2016_v02uwy10sec.csv', parse_dates=[['date', 'time']], index_col='date_time',usecols=['date','time','latitude(degree_north)','longitude(degree_east)'])
    uw = uw.resample('1D').mean()
    uw.columns = ['lat','lon']
    
    for run in runs:
        pipeline(uw, run)

# Capricorn 2
if run_cap2 == True:
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
if run_marcus == True:
    print("TRACKING: MARCUS")
    v_name = 'aa1718'
    voyages = sorted(gb.glob(in_path+'marcus_uw/201718_Voyage*.csv'))
    uw = pd.concat([pd.read_csv(v) for v in voyages ])
    uw = uw.set_index(pd.to_datetime(uw['date_time_utc']))
    uw = uw[['latitude','longitude']]
    uw = uw.resample('1D', kind='Date').mean().ffill()
    uw.columns = ['lat','lon']
    
    for run in runs:
        pipeline(uw, run)

# CAMMPCAN
if run_cammpcan == True:
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
if run_tan1802 == True:
    print("TRACKING: TAN1802")
    v_name = 'tan1718'
    uw = keycutter(in_path+'weather.csv', 'Time (UTC)', 'Latitude', 'Longitude')
    uw = uw.dropna()
    
    for run in runs:
        pipeline(uw, run)

# MQI
if run_mqi == True:
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
if run_cgo == True:
    print("TRACKING: CAPE GRIM")
    v_name = 'cgo1618'
    uw = pd.read_excel(in_path+'cgbpaps_CN  2016-2018.xlsx', index_col='date')
    uw = uw.astype(float)
    uw = uw.resample('1D', kind='date_time').mean()
    uw['lat'] = -40.68333
    uw['lon'] = 144.6833
    uw = uw[['lat','lon']]

    for run in runs:
        pipeline(uw, run)

print("DONE")