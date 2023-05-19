from MESAcontroller import ProjectOps, MesaAccess

import numpy as np
import pandas as pd
from rich import print, progress
import os, shutil
from itertools import repeat, product
from multiprocessing import Pool
import glob


import helper



def evo_star(name, mass, metallicity, v_surf_init, logging, parallel, cpu_this_process, uniform_rotation):
    produce_track = True
    print(f"Mass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s")
    ## Create working directory
    proj = ProjectOps(name)    
    initial_mass = mass
    Zinit = metallicity 
    if produce_track:
        proj.create(overwrite=True) 
        with open(f"{name}/run.log", "a+") as f:
            f.write(f"Mass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s\n")
            f.write(f"CPU: {cpu_this_process}\n\n")
        star = MesaAccess(name)
        star.load_HistoryColumns("../MESA-grid/src/templates/history_columns.list")
        star.load_ProfileColumns("../MESA-grid/src/templates/profile_columns.list")

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
                                'relax_omega_max_yrs_dt' : 1.0E-5}
        
        convergence_helper = {"convergence_ignore_equL_residuals" : True}  

        inlist_template = "../MESA-grid/src/templates/inlist_template"
        failed = True   ## Flag to check if the run failed, if it did, we retry with a different initial mass (M+dM)
        retry = 0
        total_retries = 2
        retry_type, terminate_type = None, None
        while retry<total_retries and failed:
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
                    if uniform_rotation:
                        star.set({"set_uniform_am_nu_non_rot": True}, force=True)
                    if retry > 0:
                        if retry_type == "delta_lgTeff":
                            teff_helper(star)
                        else:
                            star.set(convergence_helper, force=True)
                    if phase_name == "Pre-MS Evolution":
                        ## Initiate rotation
                        if v_surf_init>0:
                            star.set(rotation_init_params, force=True)
                        proj.run(logging=logging, parallel=parallel)
                    else:
                        proj.resume(logging=logging, parallel=parallel)
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
                    if retry == total_retries-1:
                        f.write(f"Max retries reached. Model skipped!\n")
                        break
                    f.write(f"\nMass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s\n")
                    f.write(f"Failed at phase: {phase_name}\n")
                    if retry_type == "delta_lgTeff":
                        f.write(f"Retrying with \"T_eff helper\"\n")
                    else:
                        f.write(f"Retrying with \"convergence helper\"\n")

    # profiles, gyre_input_params = get_gyre_params(name, Zinit)
    # profiles = [profile.split('/')[-1] for profile in profiles]
    # os.environ["OMP_NUM_THREADS"] = "4"
    # proj.runGyre(gyre_in="../MESA-grid/src/templates/gyre_rot_template_dipole.in", files=profiles, data_format="GYRE", 
    #             logging=False, parallel=True, n_cores=cpu_this_process, gyre_input_params=gyre_input_params)
    
def teff_helper(star):
    delta_lgTeff_limit = star.get("delta_lgTeff_limit")
    delta_lgTeff_hard_limit = star.get("delta_lgTeff_hard_limit")
    delta_lgTeff_limit += delta_lgTeff_limit/10
    delta_lgTeff_hard_limit += delta_lgTeff_hard_limit/4
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
    M = [1.22]
    Z = [0.004]
    prod = list(product(M, Z))
    M = [i[0] for i in prod]
    Z = [i[1] for i in prod]
    V = 20
    uniform_rotation = True
    length = len(M)
    n_cores = os.cpu_count()
    n_procs = length
    cpu_per_process = n_cores//n_procs
    os.environ["OMP_NUM_THREADS"] = str(cpu_per_process)
    print(f"Running {length} tracks with {n_procs} processes and {cpu_per_process} cores per process.")
    with progress.Progress(*helper.progress_columns()) as progressbar:
        task = progressbar.add_task("[b i green]Running...", total=length)
        with Pool(n_procs, initializer=helper.unmute) as pool:
            args = zip([f"test/test_M{M[i]}_Z{Z[i]}" for i in range(len(M))], M, Z, repeat(V),
                                    repeat(True), repeat(True), repeat(cpu_per_process), repeat(uniform_rotation))
            for _ in pool.istarmap(evo_star, args):
                progressbar.advance(task)

