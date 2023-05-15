from MESAcontroller import ProjectOps, MesaAccess

import numpy as np
import pandas as pd
from rich import print, progress
import os, shutil
from itertools import repeat, product
from multiprocessing import Pool
import glob


import helper



def evo_star(name, mass, metallicity, v_surf_init, logging, parallel, cpu_this_process):
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

        inlist_template = "../src/templates/inlist_template"
        failed = True   ## Flag to check if the run failed, if it did, we retry with a different initial mass (M+dM)
        retry = -1
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
                    if phase_name == "Pre-MS Evolution":
                        ## Initiate rotation
                        if v_surf_init>0:
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

    profiles, gyre_input_params = get_gyre_params(name, Zinit)
    profiles = [profile.split('/')[-1] for profile in profiles]
    os.environ["OMP_NUM_THREADS"] = "4"
    proj.runGyre(gyre_in="../src/templates/gyre_rot_template_dipole.in", files=profiles, data_format="GYRE", 
                logging=False, parallel=True, n_cores=cpu_this_process, gyre_input_params=gyre_input_params)
    

def evolve2(name, mass, metallicity, v_surf_init, logging, parallel, cpu_this_process):
    print(f"Mass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s")
    ## Create working directory
    proj = ProjectOps(name)    
    proj.create(overwrite=True) 
    with open(f"{name}/run.log", "a+") as f:
        f.write(f"Mass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s\n")
        f.write(f"CPU: {cpu_this_process}\n\n")
    star = MesaAccess(name)
    star.load_HistoryColumns("../src/templates/history_columns.list")
    star.load_ProfileColumns("../src/templates/profile_columns.list")

    initial_mass = mass
    Zinit = metallicity

    convergence_helper = {"convergence_ignore_equL_residuals" : True}  

    proj.clean()
    proj.make(silent=True)
    inlists = glob.glob("simon_inlists/*_inlist*")
    terminal_age = float(np.round(2500/initial_mass**2.5,1)*1.0E6)
    phase_max_age = [1E6, 1E7, 4.0E7, terminal_age]

    ## 1
    star.load_InlistProject(inlists.pop(0))
    Yinit, initial_h1, initial_h2, initial_he3, initial_he4 = helper.initial_abundances(Zinit)
    params = {'initial_mass': initial_mass, 'initial_z': Zinit, 'Zbase': Zinit, 'initial_y': Yinit,
        'initial_h1': initial_h1,'initial_h2': initial_h2, 
        'initial_he3': initial_he3, 'initial_he4': initial_he4, 'max_age': phase_max_age.pop(0)}
    star.set(params, force=True)
    proj.run(logging=logging, parallel=parallel)
    star.set(params, force=True)
    proj.run(logging=logging, parallel=parallel)

    ## 2
    star.load_InlistProject(inlists.pop(0))
    Yinit, initial_h1, initial_h2, initial_he3, initial_he4 = helper.initial_abundances(Zinit)
    params = {'initial_mass': initial_mass, 'relax_mass': False, 'new_mass': initial_mass, 
              'initial_z': Zinit, 'Zbase': Zinit, 'initial_y': Yinit, 'max_age': phase_max_age.pop(0)}
    star.set(params, force=True)
    proj.resume(logging=logging, parallel=parallel)

    ## 3
    star.load_InlistProject(inlists.pop(0))
    Yinit, initial_h1, initial_h2, initial_he3, initial_he4 = helper.initial_abundances(Zinit)
    params = {'initial_mass': initial_mass, 'relax_mass': False, 'new_mass': initial_mass, 
              'initial_z': Zinit, 'Zbase': Zinit, 'initial_y': Yinit, 'max_age': phase_max_age.pop(0),
              'history_interval': 15, 'profile_interval': 15}
    star.set(params, force=True)
    proj.resume(logging=logging, parallel=parallel)

    ## 4
    star.load_InlistProject(inlists.pop(0))
    Yinit, initial_h1, initial_h2, initial_he3, initial_he4 = helper.initial_abundances(Zinit)
    params = {'initial_mass': initial_mass, 'relax_mass': False, 'new_mass': initial_mass, 
              'initial_z': Zinit, 'Zbase': Zinit, 'initial_y': Yinit, 'max_age': phase_max_age.pop(0),
              'history_interval': 4, 'profile_interval': 4}
    star.set(params, force=True)
    proj.resume(logging=logging, parallel=parallel)

    ##5
    star.load_InlistProject(inlists.pop(0))
    Yinit, initial_h1, initial_h2, initial_he3, initial_he4 = helper.initial_abundances(Zinit)
    params = {'initial_mass': initial_mass, 'relax_mass': False, 'new_mass': initial_mass, 
              'initial_z': Zinit, 'Zbase': Zinit, 'initial_y': Yinit, 'max_age': phase_max_age.pop(0)}
    star.set(params, force=True)
    proj.resume(logging=logging, parallel=parallel)



    profiles, gyre_input_params = get_gyre_params(name, Zinit)
    profiles = [profile.split('/')[-1] for profile in profiles]
    os.environ["OMP_NUM_THREADS"] = "4"
    proj.runGyre(gyre_in="../src/templates/gyre_rot_template_dipole.in", files=profiles, data_format="GYRE", 
                logging=False, parallel=True, n_cores=cpu_this_process, gyre_input_params=gyre_input_params)


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
    M = [1.5]
    Z = [0.010, 0.012, 0.014, 0.016, 0.018, 0.020]
    prod = list(product(M, Z))
    M = [i[0] for i in prod]
    Z = [i[1] for i in prod]
    V = 0
    length = len(M)
    n_cores = os.cpu_count()
    n_procs = length
    cpu_per_process = n_cores//n_procs
    os.environ["OMP_NUM_THREADS"] = str(cpu_per_process)
    print(f"Running {length} tracks with {n_procs} processes and {cpu_per_process} cores per process.")
    with progress.Progress(*helper.progress_columns()) as progressbar:
        task = progressbar.add_task("[b i green]Running...", total=length)
        with Pool(n_procs, initializer=helper.unmute) as pool:
            args = zip([f"tests_here/test_M{M[i]}_Z{Z[i]}" for i in range(len(M))], M, Z, repeat(V),
                                    repeat(True), repeat(True), repeat(cpu_per_process))
            # for _ in pool.istarmap(evo_star, args):
            for _ in pool.istarmap(evolve2, args):
                progressbar.advance(task)

