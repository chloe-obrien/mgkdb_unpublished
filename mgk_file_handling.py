#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File handling script for formatting output files, getting file lists, and
reading and writing to database containing:
    get_file_list(out_dir,begin):       input GENE output directory and base filepath
                                           (nrg, energy, etc), return full list of files
                                           in directory
    get_suffixes(out_dir):            input GENE output directory, return list of run
                                           suffixes in the directory
    gridfs_put(filepath):               input filepath, upload  file to database, and
                                           return object_id of uploaded file
    gridfs_read(db_file):               input database filename, return contents of file
    upload_to_mongo
    isLinear
@author: Chloe O'Brien
"""

import config

from mgk_upload_processing import *
from mgk_post_processing import *
from ParIO import *
import numpy as np
from pymongo import MongoClient
import os
import gridfs
import re
from sshtunnel import SSHTunnelForwarder


def get_file_list(out_dir,begin):
    files_list = []

    #unwanted filetype suffixes for general list
    bad_ext = ('.ps','.png')

    #scan files in GENE output directory, ignoring files in '/in_par', and return list
    files = next(os.walk(out_dir))[2]
    for count, name in enumerate(files, start=0):
        if name.startswith(begin) and name.endswith(bad_ext) == False and not os.path.isdir('in_par'):
            file = out_dir + '/' + name
            files_list.append(file)
    return files_list


def get_suffixes(out_dir):
    suffixes = []

    #scan files in GENE output directory, find all run suffixes, return as list
    files = next(os.walk(out_dir))[2]
    for count, name in enumerate(files, start=0):
        if name.startswith('parameters_'):
            suffix = name.split('_',1)[1]
            suffix = '_' + suffix
            suffixes.append(suffix)
        elif name.lower().startswith('parameters.dat'):
            suffixes = ['.dat']
    return suffixes


def gridfs_put(filepath):
    #set directory and filepath
    file = open(filepath, 'rb')

    #connect to 'mgk_fusion' database
    db = config.database_connect

    #upload file to 'fs.files' collection
    fs = gridfs.GridFS(db)
    dbfile = fs.put(file, encoding='UTF-8', filepath=filepath)
    file.close()

    #grab '_id' for uploaded file
    object_id = str(dbfile)
    return(object_id)


def gridfs_read(filepath):
    #connect to 'mgk_fusion' database
    db =  config.database_connect

    #open 'filepath'
    fs = gridfs.GridFS(db)
    file = fs.find_one({"filepath": filepath})
    contents = file.read()
    return(contents)


def isLinear(name):
    #check parameters file for 'nonlinear' value
    if os.path.isfile(name + '/parameters'):
        par = Parameters()
        par.Read_Pars(name + '/parameters')
        pars = par.pardict
        linear = not pars['nonlinear']
        return(linear)
    #check folder name for linear
    elif name.find('linear') != -1:
        linear = True
        return(linear)
    #check folder name for nonlin
    elif name.find('nonlin') != -1:
        linear = False
        return(linear)


def isUploaded(out_dir,runs_coll):
    inDb = runs_coll.find({ "run_collection_name": out_dir })
    for run in inDb:
        runIn = run["run_collection_name"]
        return(runIn == out_dir)


### under construction ###
def update_mongo(out_dir,runs_coll):
    fields = ["user", "run_collection_name" ,"run_suffix" ,"keywords", "confidence", "codemods_id", "submitcmd_id", "parameters_id", "eqdisk_id", "efit_id", "autopar_id", "energy_id", "nrg_id", "omega_id", "scanlog_id", "scaninfo_id", "Qes", "ky", "kx", "omt", "omn", "gamma"]
    update_null = input('Would you like to update only "Null" fields?  (y/n)')
    for field in fields:
        if update_null == 'y' or update_null == 'Y':
            key = None
        key = None
        runs_coll.find({field: key })

def find_runs_in_mongo(mongo_key, key_value):
    runs_found = []
    runs_coll = config.database_connect.runs
    runs = runs_coll.find({ mongo_key: key_value })
    for run in runs:
        runs_found.append(run)
    return(runs_found)

def find_all_in_mongo():
    runs_found = []
    runs_coll = config.database_connect.runs
    runs = runs_coll.find({})
    for run in runs:
        runs_found.append(run)
    return(runs_found)

def remove_from_mongo(out_dir, runs_coll):
    #find all documents containing collection name
    inDb = runs_coll.find({ "run_collection_name": out_dir })
    for run in inDb:
        #delete all matching documents
        runs_coll.delete_one(run)


def get_object_ids(out_dir):
        #generate file lists for relevant files
        autopar_files = get_file_list(out_dir, 'autopar')
        codemods_files = get_file_list(out_dir, 'codemods')
        energy_files = get_file_list(out_dir, 'energy')
        geneerr_files = get_file_list(out_dir, 'geneerr')
        nrg_files = get_file_list(out_dir, 'nrg')
        omega_files = get_file_list(out_dir, 'omega')
        parameters_files = get_file_list(out_dir, 'parameters')
        s_alpha_files = get_file_list(out_dir, 's_alpha')
        scanlog_files = get_file_list(out_dir, 'scan.log')
        scan_info_files = get_file_list(out_dir, 'scan_info.dat')

        #list of all GENE output files to be uploaded
        output_files = autopar_files + codemods_files + energy_files + geneerr_files +            nrg_files +  omega_files + parameters_files + s_alpha_files + scanlog_files +            scan_info_files

        #upload all GENE output files to database
        object_ids = []
        for files in output_files:
            object_ids.append(gridfs_put(files))

        #map from_output files to object ids
        for i in range(0,len(output_files)):
            object_ids[i] = object_ids[i] + '    ' +  output_files[i]
        return object_ids


def upload_linear(out_dir, user, linear, confidence, input_heat, keywords):
    #connect to linear collection
    runs_coll = config.database_connect.runs

    #initialize files dictionary
    files_dict =  {'scan_id': None,
                   'scanlog_id': None,
                   'scaninfo_id': None,
                   'codemods_id': None,
                   'submit_id': None,
                   'parameters_id': None,
                   'eqdisk_id': None,
                   'efit_id': None,
                   'autopar_id': None,
                   'energy_id': None,
                   'nrg_id': None,
                   'omega_id': None,
                   's_alpha_id': None
                  }

    #generate scan_info.dat
    scan_info(out_dir)

    #update files dictionary
    object_ids = get_object_ids(out_dir)
    suffixes = get_suffixes(out_dir)
    for suffix in suffixes:
        for line in object_ids:
            if line.find('codemods' + suffix) != -1:
                files_dict['codemods_id'] = line.split()[0]
            if line.find('parameters' + suffix) != -1:
                files_dict['parameters_id'] = line.split()[0]
            if line.find('autopar' + suffix) != -1:
                files_dict['autopar_id'] = line.split()[0]
            if line.find('energy' + suffix) != -1:
                files_dict['energy_id'] = line.split()[0]
            if line.find('nrg' + suffix) != -1:
                files_dict['nrg_id'] = line.split()[0]
            if line.find('omega' + suffix) != -1:
                files_dict['omega_id'] = line.split()[0]
            if line.find('scan.log') != -1:
                files_dict['scanlog_id'] = line.split()[0]
            if line.find('scan_info.dat') != -1:
                files_dict['scaninfo_id'] = line.split()[0]
            if line.find('s_alpha' + suffix) != -1:
                files_dict['s_alpha_id'] = line.split()[0]

        #find relevant quantities from in/output
        gamma = find_omega(out_dir + '/omega' + suffix)[0]
        omega = find_omega(out_dir + '/omega' + suffix)[1]
        params = find_params(out_dir + '/parameters' + suffix)
        kx = params[0]
        ky = params[1]
        omn = params[2]
        omt = params[3]

        #metadata dictonary
        meta_dict = {"user": user,
                     "run_collection_name": out_dir,
                     "run_suffix": '' + suffix,
                     "linear" : True,
                     "keywords": keywords,
                     "confidence": confidence
                    }

        #data dictionary format for linear runs
        run_data_dict = {"gamma": gamma,
                         "omega": omega,
                         "ky": ky,
                         "kx": kx,
                         "omt": omt,
                         "omn": omn
                        }

        #combine dictionaries and upload
        run_data =  {**meta_dict, **files_dict, **run_data_dict}
        runs_coll.insert_one(run_data).inserted_id
    print('Run collection \'' + out_dir + '\' uploaded succesfully.')


def upload_nonlin(out_dir, user, linear, confidence, input_heat, keywords):
    #connect to nonlinear collection
    runs_coll = config.database_connect.runs

    #initialize files dictionary
    files_dict =  {'scan_id': None,
                   'scanlog_id': None,
                   'scaninfo_id': None,
                   'codemods_id': None,
                   'submit_id': None,
                   'parameters_id': None,
                   'eqdisk_id': None,
                   'efit_id': None,
                   'autopar_id': None,
                   'energy_id': None,
                   'nrg_id': None,
                   'omega_id': None,
                   's_alpha_id': None
                  }

    #generate scan_info.dat
    scan_info(out_dir)  ### add check for file existence

    #update files dictionary
    object_ids = get_object_ids(out_dir)
    suffixes = get_suffixes(out_dir)
    for suffix in suffixes:
        for line in object_ids:
            if line.find('codemods' + suffix) != -1:
                files_dict['codemods_id'] = line.split()[0]
            if line.find('parameters' + suffix) != -1:
                files_dict['parameters_id'] = line.split()[0]
            if line.find('autopar' + suffix) != -1:
                files_dict['autopar_id'] = line.split()[0]
            if line.find('energy' + suffix) != -1:
                files_dict['energy_id'] = line.split()[0]
            if line.find('nrg' + suffix) != -1:
                files_dict['nrg_id'] = line.split()[0]
            if line.find('omega' + suffix) != -1:
                files_dict['omega_id'] = line.split()[0]
            if line.find('scan.log') != -1:
                files_dict['scanlog_id'] = line.split()[0]
            if line.find('scan_info.dat') != -1:
                files_dict['scaninfo_id'] = line.split()[0]
            if line.find('s_alpha' + suffix) != -1:
                files_dict['s_alpha_id'] = line.split()[0]

        #find relevant quantities from in/output
        Qes = get_Qes(out_dir, suffix)
        params = find_params(out_dir + '/parameters' + suffix)
        kx = params[0]
        ky = params[1]
        omn = params[2]  #### add check n_spec for suffix
        omt = params[3]

        #metadata dictonary
        meta_dict = {"user": user,
                     "run_collection_name": out_dir,
                     "run_suffix": '' + suffix,
                     "linear" : False,
                     "keywords": keywords,
                     "confidence": confidence
                    }
        #data dictionary format for nonlinear runs
        run_data_dict = {"Qes" : Qes,
                         "ky" : ky,
                         "kx" : kx,
                         "omt" : omt,
                         "omn" : omn
                        }

        #combine dictionaries and upload
        run_data =  {**meta_dict, **files_dict, **run_data_dict}
        runs_coll.insert_one(run_data).inserted_id
    print('Run collection \'' + out_dir + '\' uploaded succesfully.')


def upload_to_mongo(out_dir, user, linear, confidence, input_heat, keywords):
    #for linear runs
    if linear:
        #connect to linear collection
        runs_coll = config.database_connect.runs
        #check if folder is already uploaded, prompt update?
        if isUploaded(out_dir, runs_coll):
            update = input('Folder exists in database.  Delete and reupload folder? (y/n) ')
            if update == 'y' or update == 'Y':
                #for now, delete and reupload instead of update - function under construction
                remove_from_mongo(out_dir, runs_coll)
                upload_linear(out_dir, user, linear, confidence, input_heat, keywords)
            else:
                print('Run collection \'' + out_dir + '\' skipped.')
        else:
            upload_linear(out_dir, user, linear, confidence, input_heat, keywords)

    #for nonlinear runs
    if not linear:
        #connect to nonlinear collection
        runs_coll = config.database_connect.runs
        #check if folder is already uploaded, prompt update?
        if isUploaded(out_dir, runs_coll):
            update = input('Folder exists in database.  Delete and reupload folder? (y/n) ')
            if update == 'y' or update == 'Y':
                #for now, delete and reupload instead of update - function under construction
                remove_from_mongo(out_dir, runs_coll)
                upload_nonlin(out_dir, user, linear, confidence, input_heat, keywords)
            else:
                print('Run collection \'' + out_dir + '\' skipped.')
        else:
            upload_nonlin(out_dir, user, linear, confidence, input_heat, keywords)

### under construction ###
def upload_big(out_dir, linear):

    field_files = get_file_list(out_dir, 'field')
    mom_files = get_file_list(out_dir, 'mom')
    vsp_files = get_file_list(out_dir, 'vsp')

    output_files = field_files + mom_files + vsp_files

    object_ids = []
    for files in output_files:
        object_ids.append(gridfs_put(files))

    #map from_output files to object ids
    for i in range(0,len(output_files)):
        object_ids[i] = object_ids[i] + '    ' +  output_files[i]

    if not linear:
        object_ids = get_object_ids(out_dir)
        field_id = None
        mom_id = None
        vsp_id = None
        suffixes = get_suffixes(out_dir)
        for suffix in suffixes:
            for line in object_ids:
                if line.find('field' + suffix) != -1:
                    field_id = line.split()[0]
                if line.find('mom' + suffix) != -1:
                    mom_id = line.split()[0]
                if line.find('vsp' + suffix) != -1:
                    vsp_id = line.split()[0]

            run_data = {"field_id": field_id,
                        "mom_id": mom_id,
                        "vsp_id": vsp_id,
                       }

            runs.update(run_data).inserted_id
