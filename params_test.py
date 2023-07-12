from MESAcontroller import ProjectOps, MesaAccess

import numpy as np
import pandas as pd
from rich import print, progress
import os, psutil
from itertools import repeat, product
import glob
import time
import itertools

from src import helper, gyre, pool
from rich import print, console

def evo_star(args):
    '''
    Run MESA evolution for a single star.
    Args:
        args (tuple): tuple of arguments
            args[0] (float): initial mass
            args[1] (float): metallicity
            args[2] (float): initial surface rotation velocity
            args[3] (str): track number
            args[4] (bool): whether to run GYRE
            args[5] (bool): whether to save the track
            args[6] (bool): whether to log the evolution in a run.log file
            args[7] (bool): whether this function is being run in parallel with ray
    '''
    dir, name, mass, metallicity, v_surf_init, param, gyre_flag, logging, parallel, cpu_this_process, produce_track, uniform_rotation = args
    os.chdir(dir)
    trace = None

    print(f"Mass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s")

    ## Create working directory
    proj = ProjectOps(name)   
    initial_mass = mass
    Zinit = metallicity

    if produce_track:  
        start_time = time.time()
        proj.create(overwrite=True) 
        with open(f"{name}/run.log", "a+") as f:
            f.write(f"Mass: {mass} MSun, Z: {metallicity}, v_init: {v_surf_init} km/s\n")
            f.write(f"CPU: {cpu_this_process}\n\n")
        star = MesaAccess(name)
        star.load_HistoryColumns("./src/templates/history_columns.list")
        star.load_ProfileColumns("./src/templates/profile_columns.list")

        rotation_init_params = {'change_rotation_flag': True,   ## False for rotation off until near zams
                                'new_rotation_flag': True,
                                'change_initial_rotation_flag': True,
                                'set_initial_surface_rotation_v': True,
                                'set_surface_rotation_v': True,
                                'new_surface_rotation_v': v_surf_init,
                                'relax_surface_rotation_v' : True,
                                'num_steps_to_relax_rotation' : 100,  ## Default value is 100
                                'relax_omega_max_yrs_dt' : 1.0E-5}   ## Default value is 1.0E9
        
        convergence_helper = {"convergence_ignore_equL_residuals" : True}  

        inlist_template = "./src/templates/inlist_template"
        failed = True   ## Flag to check if the run failed, if it did, we retry with a different initial mass (M+dM)
        retry = 0
        total_retries = 2
        retry_type, terminate_type = None, None
        while retry<=total_retries and failed:
            proj.clean()
            proj.make(silent=True)
            phases_params = helper.phases_params(initial_mass, Zinit)     
            phases_names = phases_params.keys()
            stopping_conditions = [{"stop_at_phase_ZAMS":True}, {"max_age":4e7}, {"stop_at_phase_TAMS":True}, "ERGB"]
            max_timestep = [1E4, 1E5, 2E6, 1E7]
            profile_interval = [1, 1, 5, 5]
            for phase_name in phases_names:
                try:
                    ## Run from inlist template by setting parameters for each phase
                    star.load_InlistProject(inlist_template)
                    print(phase_name)
                    star.set(phases_params[phase_name], force=True)

                    ## TEST PARAMETER
                    star.set(param, force=True)

                    ## History and profile interval
                    star.set({'history_interval':1, "profile_interval":profile_interval.pop(), "max_num_profile_models":3000})
                    
                    ##Timestep 
                    star.set({"max_years_for_timestep": max_timestep.pop(0)}, force=True)
                    
                    ## Stopping conditions
                    stopping_condition = stopping_conditions.pop(0)
                    if  stopping_condition == "ERGB":
                        ergb_params = {'Teff_lower_limit' : 6000}
                        star.set(ergb_params, force=True)
                    else:
                        star.set(stopping_condition, force=True)

                    ### Checks
                    if uniform_rotation:
                        star.set({"set_uniform_am_nu_non_rot": True}, force=True)
                    if retry > 0:
                        if "delta_lgTeff" in retry_type:
                            teff_helper(star, retry)
                        else:
                            star.set(convergence_helper, force=True)

                    ## RUN
                    if phase_name == "Evolution to ZAMS":
                        ## Initiate rotation
                        if v_surf_init>0:
                            star.set(rotation_init_params, force=True)
                        print(f"End age: {proj.run(logging=logging, parallel=parallel, trace=trace):.2e} yrs\n")
                    else:
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
    else:
        failed = False

    if gyre_flag:   ## Optional, GYRE can berun separately using the run_gyre function  
        if not failed:
            try:
                print("[bold green]Running GYRE...[/bold green]")
                os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
                os.environ['OMP_NUM_THREADS'] = '2'
                profiles, gyre_input_params = gyre.get_gyre_params(name, Zinit)
                if len(profiles) > 0:
                    profiles = [profile.split('/')[-1] for profile in profiles]
                    proj.runGyre(gyre_in=os.path.expanduser("~/workspace/MESA-grid/src/templates/gyre_rot_template_dipole.in"), files=profiles, data_format="GYRE", 
                                logging=False, parallel=True, n_cores=cpu_this_process, gyre_input_params=gyre_input_params)
                else:
                    with open(f"{name}/run.log", "a+") as f:
                        f.write(f"GYRE skipped: no profiles found, possibly because all models had T_eff < 6000 K\n")
                    gyre_flag = False
            except Exception as e:
                print("Gyre failed for track ", name)
                print(e)


def teff_helper(star, retry):
    delta_lgTeff_limit = star.get("delta_lgTeff_limit")
    delta_lgTeff_hard_limit = star.get("delta_lgTeff_hard_limit")
    # delta_lgTeff_limit += delta_lgTeff_limit/10
    delta_lgTeff_hard_limit += retry*delta_lgTeff_hard_limit
    star.set({"delta_lgTeff_limit": delta_lgTeff_limit, "delta_lgTeff_hard_limit": delta_lgTeff_hard_limit}, force=True)







if __name__ == "__main__":
    dir = os.getcwd()
    nf = "_dsct_new"
    folder = f"test{nf}"
    parallel = True
    use_ray = False
    produce_track = True
    cpu_per_process = 16

    # param_name = "mesh_delta_coeff"
    # param_range = np.arange(0.1, 1.4, 0.3)
    # param_range = np.append(param_range, [1.25])
    # param_sample = [{param_name:c} for c in param_range]

    param_sample = [{"mesh_delta_coeff":1.1}]

    M_sample = np.arange(1.5, 2.2, 0.2)
    Z_sample = np.arange(0.002, 0.023, 0.004)
    V_sample = np.arange(0, 14, 4.)
    combinations = list(itertools.product(M_sample, Z_sample, V_sample, param_sample))
    
    M = []
    Z = []
    V = []
    params = []
    names = []
    i = 1
    for m, z, v, param in combinations:
        M.append(m)
        Z.append(z)
        V.append(v)
        params.append(param)
        names.append(f"{folder}/m{m}_z{z}_v{v}_param{i}")
        i += 1

    length = len(names)
    # cpu_per_process = int(psutil.cpu_count(logical=False)//length)
    print(f"Total models: {length}\n")
    if parallel:
        args = zip(repeat(dir), names, M, Z, V, params, repeat(True), repeat(True), repeat(True), repeat(cpu_per_process), repeat(produce_track), repeat(True))
        if use_ray:
            import ray   
            try:
                ray.init(address="auto")
            except:
                ## Start the ray cluster
                with console.Console().status("[b i][blue]Starting ray cluster...[/blue]") as status:
                    pool.start_ray()
                print("[b i][green]Ray cluster started.[/green]\n")
                ray.init(address="auto")
            print("\n[b i][blue]Ray cluster resources:[/blue]")
            print("CPUs: ", ray.cluster_resources()["CPU"])
            pool.ray_pool(evo_star, args, length, cpu_per_process=cpu_per_process, initializer=helper.unmute)
        else:
            pool.mp_pool(evo_star, args, length, cpu_per_process=cpu_per_process, initializer=helper.unmute)
    else:
        os.environ["OMP_NUM_THREADS"] = '12'
        for i in range(len(names)):
            evo_star((dir, names[i], M[i], Z[i], V[i], params[i], True, False, cpu_per_process, produce_track))

 