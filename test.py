from MESAcontroller import MesaAccess, ProjectOps

import numpy as np
import pandas as pd
from rich import print, progress
import os, shutil, psutil
from itertools import repeat, product
from multiprocessing import Pool
import glob
import time


import helper



def evo_star(args):
    name, mass, metallicity, v_surf_init, logging, parallel, cpu_this_process, produce_track = args
    trace = None
    start_time = time.time()
    ## Create/Resume working directory
    proj = ProjectOps(name)    
    initial_mass = mass
    Zinit = metallicity 
    if produce_track:
        print(f"Mass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s")
        proj.create(overwrite=True) 
        with open(f"{name}/run.log", "a+") as f:
            f.write(f"Mass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s\n")
            f.write(f"CPU: {cpu_this_process}\n")
            f.write(f"OMP_NUM_THREADS: {os.environ['OMP_NUM_THREADS']}\n\n")
        star = MesaAccess(name)
        star.load_HistoryColumns("templates/history_columns.list")
        star.load_ProfileColumns("templates/profile_columns.list")
        
        overshoot_params = {'overshoot_scheme(1)': 'exponential', #exponential, step, other
                            'overshoot_zone_type(1)': 'burn_H', #burn_H, burn_He, burn_Z, nonburn, any
                            'overshoot_zone_loc(1)': 'core', #core, shell, any
                            'overshoot_bdy_loc(1)': 'top', #top, bottom, any
                            'overshoot_f(1)': 0.022, #0.022
                            'overshoot_f0(1)': 0.002, #0.002
                            'overshoot_scheme(2)': 'exponential', #exponential, step, other
                            'overshoot_zone_type(2)': 'nonburn', #burn_H, burn_He, burn_Z, nonburn, any
                            'overshoot_zone_loc(2)': 'shell', #core, shell, any
                            'overshoot_bdy_loc(2)': 'any', #top, bottom, any
                            'overshoot_f(2)': 0.006, #0.006
                            'overshoot_f0(2)': 0.001, #0.001
                            'overshoot_D_min': 1.0E-2,
                            ##mixing
                            'do_conv_premix': True}

        initial_mass = mass
        Zinit = metallicity
        rotation_init_params = {'change_rotation_flag': True,   ## False for rotation off until near zams
                                'new_rotation_flag': True,
                                'change_initial_rotation_flag': True,
                                'set_initial_surface_rotation_v': True,
                                'set_surface_rotation_v': True,
                                'new_surface_rotation_v': v_surf_init,
                                'relax_surface_rotation_v' : True,
                                'num_steps_to_relax_rotation' : 100,  ## Default value is 100
                                'relax_omega_max_yrs_dt' : 1.0E-5,   ## Default value is 1.0E9
                                'set_uniform_am_nu_non_rot': True}
        
        convergence_helper = {"convergence_ignore_equL_residuals" : True}  

        inlist_template = "templates/inlist_template"
        failed = True   ## Flag to check if the run failed, if it did, we retry with a different initial mass (M+dM)
        retry = 0
        total_retries = 2
        retry_type, terminate_type = None, None
        uniform_rotation = True
        while retry<=total_retries and failed:
            proj.clean()
            proj.make(silent=True)
            phases_params = helper.phases_params(initial_mass, Zinit)     
            phases_names = phases_params.keys()
            terminal_age = float(np.round(2500/initial_mass**2.5,1)*1.0E6)
            phase_max_age = [1E6, 1E7, 4.0E7, terminal_age]         ## 1E7 is the age when we switch to a coarser timestep
            for phase_name in phases_names:
                try:
                    ## Run from inlist template by setting parameters for each phase
                    star.load_InlistProject(inlist_template)
                    print(phase_name)
                    star.set(phases_params[phase_name], force=True)
                    star.set('max_age', phase_max_age.pop(0), force=True)
                    # star.set(overshoot_params, force=True)
                    if uniform_rotation:
                        star.set({"set_uniform_am_nu_non_rot": True}, force=True)
                    if retry > 0:
                        if "delta_lgTeff" in retry_type:
                            teff_helper(star)
                        else:
                            star.set(convergence_helper, force=True)
                    if phase_name == "Pre-MS Evolution":
                        ## Initiate rotation
                        if v_surf_init>0:
                            star.set(rotation_init_params, force=True)
                        proj.run(logging=logging, parallel=parallel, trace=trace)
                    else:
                        proj.resume(logging=logging, parallel=parallel, trace=trace)
                except Exception as e:
                    failed = True
                    print(e)
                    retry_type, terminate_type = helper.read_error(name)
                    break
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                else:
                    failed = False
            if failed:
                retry += 1
                with open(f"{name}/run.log", "a+") as f:
                    if retry == total_retries:
                        f.write(f"Max retries reached. Model skipped!\n")
                        break
                    f.write(f"\nMass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s\n")
                    f.write(f"Failed at phase: {phase_name}\n")
                    if "delta_lgTeff" in retry_type:
                        f.write(f"Retrying with \"T_eff helper\"\n")
                    else:
                        f.write(f"Retrying with \"convergence helper\"\n")
        end_time = time.time()
        with open(f"{name}/run.log", "a+") as f:
            f.write(f"Total time: {end_time-start_time} s\n\n")

    gyre = True
    if gyre:
        try:
            if not os.path.exists(f"{name}/gyre.log"):
                profiles, gyre_input_params = get_gyre_params(name, Zinit)
                profiles = [profile.split('/')[-1] for profile in profiles]
                os.environ["OMP_NUM_THREADS"] = "1"
                proj.runGyre(gyre_in="templates/gyre_rot_template_dipole.in", files=profiles, data_format="GYRE", 
                            logging=False, parallel=True, n_cores=cpu_this_process, gyre_input_params=gyre_input_params)
            else:
                print("Gyre already ran for track ", name)
        except:
            print("Gyre failed for track ", name)

def teff_helper(star):
    delta_lgTeff_limit = star.get("delta_lgTeff_limit")
    delta_lgTeff_hard_limit = star.get("delta_lgTeff_hard_limit")
    delta_lgTeff_hard_limit += delta_lgTeff_hard_limit
    star.set({"delta_lgTeff_limit": delta_lgTeff_limit, "delta_lgTeff_hard_limit": delta_lgTeff_hard_limit}, force=True)



def get_gyre_params(name, zinit):
    histfile = f"{name}/LOGS/history.data"
    pindexfile = f"{name}/LOGS/profiles.index"
    h = pd.read_csv(histfile, delim_whitespace=True, skiprows=5)
    p = pd.read_csv(pindexfile, skiprows=1, names=['model_number', 'priority', 'profile_number'], delim_whitespace=True)
    h = pd.merge(h, p, on='model_number', how='outer')
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
        if row["log_Teff"] < 3.778:
            continue
        else:
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

    

if __name__ == "__main__":
    prefix = "overshoot"
    M = np.arange(1.4, 2, 0.05)
    Z = [0.024]
    prod = list(product(M, Z))
    M = [i[0] for i in prod]
    Z = [i[1] for i in prod]
    V = 0
    uniform_rotation = True

    parallel = True
    if parallel:
        length = len(M)
        n_cores = psutil.cpu_count(logical=False)
        n_procs = length
        cpu_per_process = n_cores//n_procs
        os.environ["OMP_NUM_THREADS"] = str(cpu_per_process)
        print(f"Running {length} tracks with {n_procs} processes and {cpu_per_process} cores per process.")
        with progress.Progress(*helper.progress_columns()) as progressbar:
            task = progressbar.add_task("[b i green]Running...", total=length)
            with Pool(n_procs, initializer=helper.mute) as pool:
                args = zip([prefix+f"/test_M{M[i]}_Z{Z[i]}" for i in range(len(M))], M, Z, repeat(V),
                                        repeat(True), repeat(True), repeat(cpu_per_process), repeat(uniform_rotation))
                for _ in enumerate(pool.imap_unordered(evo_star, args)):
                    progressbar.advance(task)
    else:
        trace = ["surf_avg_v_rot", "surf_avg_omega_div_omega_crit"]
        cpu_per_process = psutil.cpu_count(logical=False)
        os.environ["OMP_NUM_THREADS"] = str(cpu_per_process)
        for i in range(len(M)):
            evo_star(f"test/test_M{M[i]}_Z{Z[i]}", M[i], Z[i], V, True, False, cpu_per_process, uniform_rotation, trace=trace)


