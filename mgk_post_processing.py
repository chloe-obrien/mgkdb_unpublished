#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Post processing script containing:
    get_nrg(out_dir, suffix):               input put
    get_Qes(filepath):                      input 'nrg' filepath return saturated Qes for
                                               nonlinear runs
    find_params(filepath):                  input 'parameters' filepath, return wanted values
    find_omega(filepath):                   input 'omega' filepath, return omega, gamma values
                                               for linear runs
    get_scanlog(filepath):                  input scan.log file, return scan parameter, growth 
                                               rate, and frequency for linear runs
    get_quasilinear(filepath):              ***in development***
    get_omega_from_field(out_dir, suffix):  input output directory and run suffix, return
                                               omega, gamma values for nonlinear runs
                                               ***in development***
    plot_linear(out_dir,scan_param,freq):   input output directory, scan parameter, and desired
                                               frequency, saves plot
                                               possible scan_param: 'kx', 'ky', 'TiTe', 'omn', 'omt'
                                               possible freq: 'gamma', 'omega'
@author: Chloe O'Brien
"""

from mgk_upload_processing import *
from mgk_file_handling import *
import math
import numpy as np
import optparse as op
import matplotlib.pyplot as plt
from fieldlib import *
from ParIO import * 
from finite_differences import *
from sys import path
from sys import exit
import os


def mean_gamma(mongo_key, key_value):
    gamma = []
    all_gamma = []
    runs = find_runs_in_mongo(mongo_key, key_value)
    for run in runs:
        all_gamma.append(run['gamma'])
        
    for i in range(0,len(all_gamma)):
        if type(all_gamma[i]) == str or math.isnan(all_gamma[i]):
            pass
        else:
            gamma.append(all_gamma[i])
        print(gamma)
        av_gamma = sum(gamma) / len(gamma)
    return(av_gamma)

def gamma_vs_om(run):
    all_gamma = []
    all_omt = []
    all_omn = []
    runs = find_runs_in_mongo('linear',False)
    for run in runs:
        all_gamma.append(run['gamma'])
        all_omt.append(run['omt']) 
        all_omn.append(run['omn']) 
    return

def get_distributions():
       
        return

def quasilinear():
    all_gamma = []
    all_kx = []
    all_ky = []
    gamma = []
    kx = []
    ky = []

    runs = find_runs_in_mongo('linear',True)
    for run in runs:
        all_gamma.append(run['gamma'])
        all_kx.append(run['kx'])
        all_ky.append(run['ky'])
        
    for i in range(0,len(all_kx)):
        if type(all_kx[i]) == str or type(all_ky[i]) == str or type(all_gamma[i]) == str:
            pass
        else:
            gamma.append(all_gamma[i])
            kx.append(all_kx[i])
            ky.append(all_ky[i])
        
    sum_kx = 0
    sum_ky = 0
    
    for kx in kx:
        sum_kx = sum_kx + kx**2
    for ky in ky:
        sum_ky = sum_ky + ky**2    
    av_kx = np.sqrt(sum_kx)/len(all_kx)
    av_ky = np.sqrt(sum_ky)/len(all_ky)
    av_gamma = sum(gamma)/len(gamma)
    
    D_ml = av_gamma / av_kx
    D_ml_other = av_gamma / av_ky
    D_ml_ext = av_gamma / (av_kx + av_ky)
    
    return(D_ml, D_ml_ext)

def find_nan_gamma():
    db =  config.database_connect.runs
    
    runs = find_runs_in_mongo('gamma', np.nan)
    for run in runs:
        out_dir = run['run_collection_name']
        suffix = run['run_suffix']
        new_gamma = calc_gamma(out_dir, suffix, True)
        db.update_one({'_id': run['_id']},{'$set': {'gamma': new_gamma}})
        