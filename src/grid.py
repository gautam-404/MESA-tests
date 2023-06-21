import os
from itertools import repeat

import numpy as np
from rich import print

from . import helper
from .mesa import evo_star
from .pool import mp_pool, ray_pool


def init_grid(mass_range=None, metallicity_range=None, v_surf_init_range=None, load_grid=False, filepath=None, save_grid=None):
    '''
    Initialize the grid
    Args:   
        mass_range (optional, tuple or list): range of masses, of the form (min, max, step)
        metallicity_range (optional, tuple or list): range of metallicities, of the form (min, max, step)
        v_surf_init_range (optional, tuple or list): range of initial surface velocities, of the form (min, max, step)
        create_grid (optional, bool): whether to create the grid from scratch
    Returns:
            masses (list): list of masses
            metallicities (list): list of metallicities
            v_surf_init_list (list): list of initial surface velocities
    '''
    def get_grid(sample_masses, sample_metallicities, sample_v_init):
        '''
        Get the grid from the sample lists
        '''
        masses = np.repeat(sample_masses, len(sample_metallicities)*len(sample_v_init)).astype(float) 
        metallicities = np.tile(np.repeat(sample_metallicities, len(sample_v_init)), len(sample_masses)).astype(float)
        v_surf_init_list = np.tile(sample_v_init, len(sample_masses)*len(sample_metallicities)).astype(float) 
        # ## Uncomment to print grid
        # print(list(map(tuple, np.dstack(np.array([masses, metallicities, v_surf_init_list]))[0])))
        # print(len(masses), len(metallicities), len(v_surf_init_list))
        # exit()    
        return masses, metallicities, v_surf_init_list
    
    def check_ranges(mass_range, metallicity_range, v_surf_init_range):
        '''
        Check if the ranges are valid
        '''
        if mass_range is None and metallicity_range is None and v_surf_init_range is None:
            return True, None
        elif mass_range is not None and metallicity_range is not None and v_surf_init_range is not None:
            return True, True
        else:
            return False, False

    range_check, value_check = check_ranges(mass_range, metallicity_range, v_surf_init_range)
    if range_check is True:
        if value_check is True:
            sample_masses = np.arange(mass_range[0], mass_range[1], mass_range[2])               
            sample_metallicities = np.arange(metallicity_range[0], metallicity_range[1], metallicity_range[2])
            sample_v_init = np.arange(v_surf_init_range[0], v_surf_init_range[1], v_surf_init_range[2])
            masses, metallicities, v_surf_init_list = get_grid(sample_masses, sample_metallicities, sample_v_init)
        else:
            sample_masses = np.arange(1.2, 2.22, 0.02)                ## 1.36 - 2.20 Msun (0.02 Msun step)
            sample_metallicities = np.arange(0.001, 0.027, 0.001)    ## 0.001 - 0.012 (0.0001 step)
            sample_v_init = np.arange(0, 22, 2)                          ## 0 - 20 km/s (2 km/s step)
            masses, metallicities, v_surf_init_list = get_grid(sample_masses, sample_metallicities, sample_v_init)
    else:
        raise("Invalid range. Please provide all ranges or none.")

    if load_grid:
        if filepath is None:
            filepath = os.path.abspath("./src/templates/coarse_age_map.csv")
        else:
            filepath = os.path.abspath(filepath)
            if not os.path.exists(filepath):
                raise("File not found.")
        ## Load grid
        arr = np.genfromtxt(filepath,
                        delimiter=",", dtype=str, skip_header=1)
        masses = arr[:,0].astype(float)
        metallicities = arr[:,1].astype(float)
        v_surf_init_list = np.random.randint(1, 10, len(masses)).astype(float) * 30

    if save_grid is not None and save_grid is not False:
        if isinstance(save_grid, str):
            name = save_grid
        else:
            name = "track_index.dat"
        np.savetxt(name, np.dstack(np.array([np.arange(1, len(masses)+1, 1), masses, metallicities, v_surf_init_list]))[0], 
           fmt="%i\t\t%.2f\t\t%.3f\t\t%.1f")
    return masses, metallicities, v_surf_init_list


def run_grid(masses, metallicities, v_surf_init_list, tracks_list=None, cpu_per_process=16, gyre_flag=False, 
            save_track=True, logging=True, overwrite=None, slice_start=None, use_ray=False, incomplete_models_list=None,
            uniform_rotation=True, config=None):
    '''
    Run the grid of tracks.
    Args:
        masses (list): list of initial masses
        metallicities (list): list of metallicities
        v_surf_init_list (list): list of initial surface velocities
        tracks_list (optional, list): track numbers corresponding to the grid points. 
                                    If None, track numbers will be automatically assigned.
        cpu_per_process (optional, int): number of CPUs to use per process
        gyre (optional, bool): whether to run GYRE on the tracks
        save_track (optional, bool): whether to save the track after the run
        logging (optional, bool): whether to log the run
        overwrite (optional, bool): whether to overwrite existing "gridwork" and "gridwork" directory. 
                                    If False, the existing "grid_archive" directory will be renamed to "grid_archive_old".
                                    If False, the existing "gridwork" will be overwritten nonetheless.
    '''

    ## Create archive directories
    helper.create_grid_dirs(overwrite=overwrite)

    ## Run grid ##
    length = len(masses)
    if tracks_list is None:
        tracks_list = range(1, length+1)
    if incomplete_models_list is not None:
        tracks_list = incomplete_models_list
        slice_start = 0
    args = zip(masses, metallicities, v_surf_init_list, tracks_list,
                    repeat(gyre_flag), repeat(save_track), repeat(logging), 
                    repeat(True), repeat(cpu_per_process), repeat(slice_start), repeat(uniform_rotation))
    
    if not use_ray:
        mp_pool(evo_star, args, length, cpu_per_process=cpu_per_process, config=config)
    else:
        ray_pool(evo_star, args, length, cpu_per_process=cpu_per_process, config=config)