import os
import shutil
import pandas as pd
import numpy as np
import glob
import tarfile
from itertools import repeat
import time
import traceback
import logging

from MESAcontroller import ProjectOps, MesaAccess
from rich import print

from .pool import ray_pool, mp_pool
from . import helper

def get_gyre_params(name, zinit=None):
    if zinit is None:
        star = MesaAccess(name)
        zinit = star.get("Zbase")
    histfile = f"{name}/LOGS/history.data"
    pindexfile = f"{name}/LOGS/profiles.index"
    h = pd.read_csv(histfile, delim_whitespace=True, skiprows=5)
    p = pd.read_csv(pindexfile, skiprows=1, names=['model_number', 'priority', 'profile_number'], delim_whitespace=True)
    h = pd.merge(h, p, on='model_number', how='right')
    h["Zfrac"] = 1 - h["average_h1"] - h["average_he4"]
    h["Myr"] = h["star_age"]*1.0E-6
    h["density"] = h["star_mass"]/np.power(10,h["log_R"])**3
    gyre_start_age = 2.0E6
    gyre_intake = h.query(f"Myr > {gyre_start_age/1.0E6}")
    profiles = []
    min_freqs = []
    max_freqs = []
    for i,row in gyre_intake.iterrows():
        p = int(row["profile_number"])

        # don't run gyre on very cool models (below about 6000 K)
        # if row["log_Teff"] < 3.778:
        #     continue
        # else:
        profiles.append(f"{name}/LOGS/profile{p}.data.GYRE")
        try:
            muhz_to_cd = 86400/1.0E6
            mesa_dnu = row["delta_nu"]
            dnu = mesa_dnu * muhz_to_cd
            freq_min = int(1.5 * dnu)
            freq_max = int(12 * dnu)
        except:
            dnu = None
            freq_min = 15
            if zinit < 0.003:
                freq_max = 150
            else:
                freq_max = 95
        min_freqs.append(freq_min)
        max_freqs.append(freq_max)
    gyre_input_params = []
    for i in range(len(profiles)):
        gyre_input_params.append({"freq_min": min_freqs[i], "freq_max": max_freqs[i]})
    return profiles, gyre_input_params



def gyre_parallel(args):
    '''Run GYRE on a tar.gz track'''
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
    track, dir_name, gyre_in, cpu_per_process = args
    track = track.split("/")[-1]
    tracks_archive = os.path.abspath(f"{dir_name}/tracks")
    gyre_archive = os.path.abspath(f"{dir_name}/gyre/freqs_{track.split('.')[0]}")
    gyre_flag = False
    try:
        with helper.cwd(tracks_archive):
            with tarfile.open(track, "r:gz") as tar:
                tar.extractall()
            name = track.split(".")[0].replace("track", "work")
            work_dir = os.path.abspath(os.path.join(tracks_archive, name))
            print(f"[b][blue]Running GYRE on[/blue] {name}")
            ## Get GYRE params
            profiles, gyre_input_params = get_gyre_params(name)
            profiles = [profiles.split("/")[-1] for profiles in profiles]
            # Run GYRE
            proj = ProjectOps(name)
            os.environ['OMP_NUM_THREADS'] = '4'

            if len(profiles) > 0:
                ## Run GYRE on multiple profile files parallely
                proj.runGyre(gyre_in=gyre_in, files=profiles, gyre_input_params=gyre_input_params, 
                            data_format="GYRE", logging=False, parallel=True, n_cores=cpu_per_process)
                gyre_flag = True
            else:
                with open(f"{name}/run.log", "a+") as f:
                    f.write(f"GYRE skipped: no profiles found, possibly because models where T_eff < 6000 K\n")
    except Exception as e:
        print(f"[b][red]Error running GYRE on[/red] {name}")
        logging.error(traceback.format_exc())
        raise e
    finally:
        if gyre_flag:
            ## Archive GYRE output
            os.mkdir(gyre_archive)
            for file in glob.glob(os.path.join(work_dir, "LOGS/*-freqs.dat")):
                shutil.copy(file, gyre_archive)
            ## Compress GYRE output
            compressed_file = f"{gyre_archive}.tar.gz"
            with tarfile.open(compressed_file, "w:gz") as tarhandle:
                tarhandle.add(gyre_archive, arcname=os.path.basename(gyre_archive))
            ## Remove GYRE output
            shutil.rmtree(gyre_archive)
            ## Remove work directory
            for i in range(5):              ## Try 5 times, then give up. NFS is weird. Gotta wait and retry.
                os.system(f"rm -rf {work_dir} > /dev/null 2>&1")
                time.sleep(0.5)             ## Wait for the process, that has the nfs files open, to die/diconnect



def run_gyre(dir_name, gyre_in, cpu_per_process=16, slice_start=0, slice_end=100000000, use_ray=False, 
            incomplete_gyre_list=None):
    '''
    Run GYRE on all tracks in the archive. OR run GYRE on a single track.
    Args:       
        dir_name (str): archive directory name
        parallel (optional, bool): whether to parallelize the evolution
    '''
    tracks_archive = os.path.abspath(f"{dir_name}/tracks/")
    gyre_archive = os.path.abspath(f"{dir_name}/gyre/")
    gyre_in = os.path.abspath(gyre_in)
    tracks = glob.glob(os.path.join(tracks_archive, "*.tar.gz"))
    if incomplete_gyre_list is not None:
        tracks = [track for track in tracks if int(track.split(".")[-3].split("_")[-1]) in incomplete_gyre_list]
    else:
        tracks = [track for track in tracks if slice_start <= int(track.split(".")[-3].split("_")[-1])-1 <= slice_end]
    args = zip(tracks, repeat(dir_name), repeat(gyre_in), repeat(cpu_per_process))
    length = len(tracks)
    try:
        if use_ray:
            ray_pool(gyre_parallel, args, length, cpu_per_process=cpu_per_process)
        else:
            mp_pool(gyre_parallel, args, length, cpu_per_process=cpu_per_process)
    except KeyboardInterrupt:
        print("[b][red]GYRE interrupted. Cleaning up.")
        # os.system(f"rm -rf {gyre_archive}/* > /dev/null 2>&1") ## Can skip this while cleaning up
        tmp = os.path.join(tracks_archive, "work*")
        os.system(f"rm -rf {tmp} > /dev/null 2>&1") ## one of the folders might not be deleted... -_-
        print("[b][red]GYRE stopped.")
        raise KeyboardInterrupt
    except Exception as e:
        print("[b][red]GYRE run failed. Check run logs.")
        raise e