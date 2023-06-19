from MESAcontroller import ProjectOps, MesaAccess

import numpy as np
import pandas as pd
from rich import print, progress
import os, psutil
from itertools import repeat, product
from multiprocessing import Pool
import glob
import time
import itertools


import helper



def evo_star(args):
    name, mass, metallicity, v_surf_init, net, logging, parallel, cpu_this_process, produce_track = args
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
            phase_max_age = [1E6, 1E7, 4.0E7, "TAMS", "ERGB"]         ## 1E7 is the age when we switch to a coarser timestep
            for phase_name in phases_names:
                try:
                    ## Run from inlist template by setting parameters for each phase
                    star.load_InlistProject(inlist_template)
                    print(phase_name)
                    star.set(phases_params[phase_name], force=True)
                    max_age = phase_max_age.pop(0)
                    if isinstance(max_age, float):
                        star.set('max_age', max_age, force=True)
                    elif max_age == "TAMS":
                        tams_params = {'xa_central_lower_limit_species(1)' : 'h1',
                                       "xa_central_lower_limit(1)" : 0.01}
                        star.set(tams_params, force=True)
                    elif max_age == "ERGB":
                        ergb_params = {'Teff_lower_limit' : 5000}
                        star.set(ergb_params, force=True)
                    star.set(net, force=True)
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
                        print(f"End age: {proj.run(logging=logging, parallel=parallel, trace=trace):.2e} yrs\n")
                    else:
                        # if phase_name == "Late Main Sequence Evolution":
                        #     print("Phase skipped")
                        #     continue
                        print(f"End age: {proj.resume(logging=logging, parallel=parallel, trace=trace):.2e} yrs\n")
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
            # if not os.path.exists(f"{name}/gyre.log"):
            profiles, gyre_input_params = get_gyre_params(name, Zinit)
            profiles = [profile.split('/')[-1] for profile in profiles]
            os.environ["OMP_NUM_THREADS"] = "1"
            proj.runGyre(gyre_in="templates/gyre_rot_template_dipole.in", files=profiles, data_format="GYRE", 
                        logging=False, parallel=True, n_cores=cpu_this_process, gyre_input_params=gyre_input_params)
            # else:
            #     print("Gyre already ran for track ", name)
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
    gyre_start_age = 1e6
    gyre_intake = h.query(f"Myr > {gyre_start_age/1.0E6}")
    profiles = []
    min_freqs = []
    max_freqs = []
    for i,row in gyre_intake.iterrows():
        p = int(row["profile_number"])

        ## don't run gyre on very cool models (below about 6000 K)
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

    

if __name__ == "__main__":
    nets_sample = [{'change_net' : True, 'new_net_name' : 'basic.net',
                    'change_initial_net' : True, 'adjust_abundances_for_new_isos' : True,
                    'show_net_species_info' : False, 'show_net_reactions_info' : False},
            {'change_net' : True, 'new_net_name' : 'pp_extras.net',
                    'change_initial_net' : True, 'adjust_abundances_for_new_isos' : True,
                    'show_net_species_info' : False, 'show_net_reactions_info' : False},
            {'change_net' : True, 'new_net_name' : 'hot_cno.net',
                    'change_initial_net' : True, 'adjust_abundances_for_new_isos' : True,
                    'show_net_species_info' : False, 'show_net_reactions_info' : False},
            {'change_net' : True, 'new_net_name' : 'pp_and_cno_extras.net',  
                    'change_initial_net' : False, 'adjust_abundances_for_new_isos' : True,
                    'show_net_species_info' : False, 'show_net_reactions_info' : False},
            {'change_net' : True, 'new_net_name' : 'pp_and_hot_cno.net',  
                    'change_initial_net' : False, 'adjust_abundances_for_new_isos' : True,
                    'show_net_species_info' : False, 'show_net_reactions_info' : False}]
    M_sample = [1.7]
    # M_sample = [1.4]
    Z_sample = [0.015]
    V_sample = [0]
    combinations = list(itertools.product(M_sample, Z_sample, V_sample, nets_sample))
    
    M = []
    Z = []
    V = []
    nets = []
    names = []
    for m, z, v, net in combinations:
        M.append(m)
        Z.append(z)
        V.append(v)
        nets.append(net)
        names.append(f"test/m{m}_z{z}_v{v}_net{net['new_net_name'].split('.')[0]}")

    length = len(nets)
    n_cores = psutil.cpu_count(logical=False)
    n_procs = length
    cpu_per_process = n_cores//n_procs
    os.environ["OMP_NUM_THREADS"] = str(cpu_per_process)
    parallel = True
    produce_track = True
    if parallel:
        print(f"Total {length} tracks.")
        print(f"Running {n_procs} processes in parallel.")
        print(f"Using {cpu_per_process} cores per process.")
        with progress.Progress(*helper.progress_columns()) as progressbar:
            task = progressbar.add_task("[b i green]Running...", total=length)
            with Pool(n_procs, initializer=helper.unmute) as pool:
                args = zip(names, M, Z, V, nets, repeat(True), repeat(True), repeat(cpu_per_process), repeat(produce_track))
                for _ in enumerate(pool.imap_unordered(evo_star, args)):
                    progressbar.advance(task)
    else:
        os.environ["OMP_NUM_THREADS"] = '4'
        for i in range(len(nets)):
            evo_star((names[i], M[i], Z[i], V[i], nets[i], True, False, cpu_per_process, produce_track))
            os.chdir("/Users/anujgautam/Documents/MESA-workspace/MESA-tests/")
