from MESAcontroller import ProjectOps, MesaAccess

import numpy as np
import pandas as pd
from rich import print, progress
import os, shutil
from itertools import repeat
from multiprocessing import Pool
import glob


import helper


def evo_star(test_profiles_folder, name, mass, metallicity, v_surf_init,
                logging, parallel, cpu_this_process, uniform_rotation=True):
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
        star.load_HistoryColumns("../src/templates/history_columns.list")
        star.load_ProfileColumns("../src/templates/profile_columns.list")
        rotation_init_params = {'change_rotation_flag': True,   ## False for rotation off until near zams
                                'new_rotation_flag': True,
                                'change_initial_rotation_flag': True,
                                'set_initial_surface_rotation_v': True,
                                'set_surface_rotation_v': True,
                                'new_surface_rotation_v': v_surf_init,
                                'relax_initial_surface_rotation_v' : True,
                                'relax_omega_max_yrs_dt' : 1.0E-5,
                                'num_steps_to_relax_rotation' : 100, ## Default value is 100
                                'set_uniform_am_nu_non_rot': True}

        convergence_helper = {"convergence_ignore_equL_residuals" : True}  ## Uses max resid dlnE_dt instead

        inlist_template = "../src/templates/inlist_template"
        failed = True   ## Flag to check if the run failed, if it did, we retry with age different initial mass (M+dM)
        retry = 0
        dM = [0, 1e-3, -1e-3, 2e-3, -2e-3]
        while retry<len(dM) and failed:
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
                    if phase_name == "Pre-MS Evolution":
                        ## Initiate rotation
                        star.set(rotation_init_params, force=True)
                        if retry>=0:
                            star.set(convergence_helper, force=True)
                        proj.run(logging=logging, parallel=parallel)
                    else:
                        if phase_name == "Late Main Sequence Evolution":
                            continue
                        proj.resume(logging=logging, parallel=parallel)
                except Exception as e:
                    failed = True
                    print(e)
                    break
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                else:
                    failed = False
            if failed:
                retry += 1
                initial_mass = mass + dM[retry]
                with open(f"{name}/run.log", "a+") as f:
                    if retry == len(dM)-1:
                        f.write(f"Max retries reached. Model skipped!\n")
                        break
                    f.write(f"\nMass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s\n")
                    f.write(f"Failed at phase: {phase_name}\n")
                    f.write(f"Retrying with dM = {dM[retry]}\n")
                    f.write(f"New initial mass: {initial_mass}\n")

    highest_density_profile = sorted(glob.glob(f"{name}/LOGS/profile*.data.GYRE"), 
                                key=lambda x: int(os.path.basename(x).split('.')[0].split('profile')[1]))[-1]
    highest_density_profile = highest_density_profile.split('/')[-1]
    profiles, gyre_input_params = get_gyre_params(name, Zinit)
    for i in range(len(profiles)):
        if highest_density_profile in profiles[i]:
            gyre_input_params = gyre_input_params[i]
            break
    
    print(f"Running GYRE on {highest_density_profile} with params: {gyre_input_params}")
    proj.runGyre(gyre_in="../src/templates/gyre_rot_template_dipole.in", files=highest_density_profile, data_format="GYRE", 
                        logging=True, parallel=False, gyre_input_params=gyre_input_params)
    os.mkdir(f"{test_profiles_folder}/V_{v_surf_init}")
    shutil.copy(f"{name}/LOGS/{highest_density_profile}", f"{test_profiles_folder}/V_{v_surf_init}/")
    shutil.copy(f"{name}/LOGS/history.data", f"{test_profiles_folder}/V_{v_surf_init}/")
    shutil.copy(f"{name}/LOGS/profiles.index", f"{test_profiles_folder}/V_{v_surf_init}/")
    shutil.copy(f"{name}/LOGS/{highest_density_profile.split('.')[0]}-freqs.dat", f"{test_profiles_folder}/V_{v_surf_init}/")
    shutil.copy(f"{name}/run.log", f"{test_profiles_folder}/V_{v_surf_init}/")
    shutil.rmtree(name)

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
    # for M in [1.20, 1.24, 1.28, 1.32, 1.36, 1.40, 1.44, 1.48, 1.52, 1.56]:
    #     for Z in [0.002, 0.004, 0.008, 0.010, 0.012]:
    #         V = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]

    uniform_rotation = True
    for M in [1.4]:
        for Z in [0.008]: 
            V = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            test_profiles_folder = f"tests_here/test_profiles_M{M}_Z{Z}"
            if os.path.exists(test_profiles_folder):
                shutil.rmtree(test_profiles_folder)
            os.mkdir(test_profiles_folder)
            length = len(V)
            n_cores = os.cpu_count()
            n_procs = length
            cpu_per_process = n_cores//n_procs
            os.environ["OMP_NUM_THREADS"] = str(cpu_per_process)
            with progress.Progress(*helper.progress_columns()) as progressbar:
                task = progressbar.add_task("[b i green]Running...", total=length)
                with Pool(n_procs, initializer=helper.mute) as pool:
                    args = zip(repeat(test_profiles_folder), [f"tests_here/track{i}" for i in range(1, length+1)], repeat(M), repeat(Z), V,
                                repeat(True), repeat(True), repeat(cpu_per_process), repeat(uniform_rotation))
                    for _ in pool.istarmap(evo_star, args):
                        progressbar.advance(task)