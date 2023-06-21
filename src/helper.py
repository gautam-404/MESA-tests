import os
import sys
import glob
import shutil
import tarfile
import numpy as np
from rich import progress, live, console, panel, prompt, print
from contextlib import contextmanager
import pandas as pd
from scipy.stats import linregress


Y_sun_phot = 0.2485 # Asplund+2009
Y_sun_bulk = 0.2703 # Asplund+2009
Z_sun_phot = 0.0134 # Asplund+2009
Z_sun_bulk = 0.0142 # Asplund+2009
Y_recommended = 0.28 # typical acceptable value, according to Joel Ong TSC2 talk.
dY_by_dZ = 1.4
h2_to_h1_ratio = 2.0E-05
he3_to_he4_ratio = 1.66E-04

dt_limit_values = ['burn steps', 'Lnuc', 'Lnuc_cat', 'Lnuc_H', 'Lnuc_He', 'lgL_power_phot', 'Lnuc_z', 'bad_X_sum',
                  'dH', 'dH/H', 'dHe', 'dHe/He', 'dHe3', 'dHe3/He3', 'dL/L', 'dX', 'dX/X', 'dX_nuc_drop', 'delta mdot',
                  'delta total J', 'delta_HR', 'delta_mstar', 'diff iters', 'diff steps', 'min_dr_div_cs', 'dt_collapse',
                  'eps_nuc_cntr', 'error rate', 'highT del Ye', 'hold', 'lgL', 'lgP', 'lgP_cntr', 'lgR', 'lgRho', 'lgRho_cntr',
                  'lgT', 'lgT_cntr', 'lgT_max', 'lgT_max_hi_T', 'lgTeff', 'dX_div_X_cntr', 'lg_XC_cntr', 'lg_XH_cntr', 
                  'lg_XHe_cntr', 'lg_XNe_cntr', 'lg_XO_cntr', 'lg_XSi_cntr', 'XC_cntr', 'XH_cntr', 'XHe_cntr', 'XNe_cntr',
                  'XO_cntr', 'XSi_cntr', 'log_eps_nuc', 'max_dt', 'neg_mass_frac', 'adjust_J_q', 'solver iters', 'rel_E_err',
                  'varcontrol', 'max increase', 'max decrease', 'retry', 'b_****']

def initial_abundances(Zinit):
    """
    Input: Zinit
    Output: Yinit, initial_h1, initial_h2, initial_he3, initial_he4
    """
    dZ = np.round(Zinit - Z_sun_bulk,4)
    dY = dY_by_dZ * dZ
    Yinit = np.round(Y_recommended + dY,4)
    Xinit = 1 - Yinit - Zinit

    initial_h2 = h2_to_h1_ratio * Xinit
    initial_he3= he3_to_he4_ratio * Yinit
    initial_h1 = (1 - initial_h2) * Xinit
    initial_he4= (1 - initial_he3) * Yinit

    return Yinit, initial_h1, initial_h2, initial_he3, initial_he4


def phases_params(initial_mass, Zinit):
    '''
    Input: initial_mass, Zinit
    Output: phases_params
    '''
    Yinit, initial_h1, initial_h2, initial_he3, initial_he4 = initial_abundances(Zinit)

    params = { 'Pre-MS Evolution':
                    {'initial_mass': initial_mass, 'initial_z': Zinit, 'Zbase': Zinit, 'initial_y': Yinit,
                    'initial_h1': initial_h1,'initial_h2': initial_h2, 
                    'initial_he3': initial_he3, 'initial_he4': initial_he4,
                    'create_pre_main_sequence_model': True, 'pre_ms_T_c': 9e5,
                    'set_initial_model_number' : True, 'initial_model_number' : 0,
                    'set_uniform_initial_composition' : True, 'initial_zfracs' : 6,
                    'change_net' : True, 'new_net_name' : 'pp_and_cno_extras.net',  
                    'change_initial_net' : False, 'adjust_abundances_for_new_isos' : True,
                    'show_net_species_info' : False, 'show_net_reactions_info' : False,
                    'relax_mass' : True, 'lg_max_abs_mdot' : 6, 'new_mass' : initial_mass,
                    'write_header_frequency': 10, 'history_interval': 1, 'terminal_interval': 10, 'profile_interval': 15,
                    'max_years_for_timestep' : 1.3e4, 
                    'delta_lgTeff_limit' : 0.005, 'delta_lgTeff_hard_limit' : 0.01,
                    'delta_lgL_limit' : 0.02, 'delta_lgL_hard_limit' : 0.05,
                    'okay_to_reduce_gradT_excess' : True, 'scale_max_correction' : 0.1,
                    'num_trace_history_values': 7,
                    'trace_history_value_name(1)': 'surf_avg_v_rot',
                    'trace_history_value_name(2)': 'surf_avg_omega_div_omega_crit',
                    'trace_history_value_name(3)': 'log_total_angular_momentum',
                    'trace_history_value_name(4)': 'surf_escape_v',
                    'trace_history_value_name(5)': 'log_g',
                    'trace_history_value_name(6)': 'log_R',
                    'trace_history_value_name(7)': 'star_mass'},

                'MS Evolution' :
                    {'Zbase': Zinit, 'change_initial_net' : False, 'show_net_species_info' : False, 'show_net_reactions_info' : False,
                    'max_years_for_timestep' : 0.75e6, 
                    'delta_lgTeff_limit' : 0.00015, 'delta_lgTeff_hard_limit' : 0.0015,
                    'delta_lgL_limit' : 0.0005, 'delta_lgL_hard_limit' : 0.005,
                    'write_header_frequency': 10, 'history_interval': 4, 'terminal_interval': 10, 'profile_interval': 4,
                    'num_trace_history_values': 7, 
                    'trace_history_value_name(1)': 'surf_avg_v_rot',
                    'trace_history_value_name(2)': 'surf_avg_omega_div_omega_crit',
                    'trace_history_value_name(3)': 'log_total_angular_momentum',
                    'trace_history_value_name(4)': 'surf_escape_v',
                    'trace_history_value_name(5)': 'log_g',
                    'trace_history_value_name(6)': 'log_R',
                    'trace_history_value_name(7)': 'star_mass'},

                'Evolution to TAMS' :
                    {'Zbase': Zinit, 'change_initial_net' : False, 'show_net_species_info' : False, 'show_net_reactions_info' : False,
                    'max_years_for_timestep' : 1e7,
                    'delta_lgTeff_limit' : 0.0006, 'delta_lgTeff_hard_limit' : 0.006,
                    'delta_lgL_limit' : 0.002, 'delta_lgL_hard_limit' : 0.02,
                    'write_header_frequency': 10, 'history_interval': 1, 'terminal_interval': 10, 'profile_interval': 1,
                    'num_trace_history_values': 7,
                    'trace_history_value_name(1)': 'surf_avg_v_rot',
                    'trace_history_value_name(2)': 'surf_avg_omega_div_omega_crit',
                    'trace_history_value_name(3)': 'log_total_angular_momentum',
                    'trace_history_value_name(4)': 'surf_escape_v',
                    'trace_history_value_name(5)': 'log_g',
                    'trace_history_value_name(6)': 'log_R',
                    'trace_history_value_name(7)': 'star_mass'},

                'Evolution post-MS' :
                    {'Zbase': Zinit, 'change_initial_net' : False, 'show_net_species_info' : False, 'show_net_reactions_info' : False,
                    'max_years_for_timestep' : 1e7,
                    'delta_lgTeff_limit' : 0.0006, 'delta_lgTeff_hard_limit' : 0.006,
                    'delta_lgL_limit' : 0.002, 'delta_lgL_hard_limit' : 0.02,
                    'write_header_frequency': 10, 'history_interval': 1, 'terminal_interval': 10, 'profile_interval': 1,
                    'num_trace_history_values': 7,
                    'trace_history_value_name(1)': 'surf_avg_v_rot',
                    'trace_history_value_name(2)': 'surf_avg_omega_div_omega_crit',
                    'trace_history_value_name(3)': 'log_total_angular_momentum',
                    'trace_history_value_name(4)': 'surf_escape_v',
                    'trace_history_value_name(5)': 'log_g',
                    'trace_history_value_name(6)': 'log_R',
                    'trace_history_value_name(7)': 'star_mass'},
    }

    return params

def mute():
    sys.stdout = open(os.devnull, 'w') 

def unmute():
    sys.stdout = sys.__stdout__

def process_outline(outline, age):
    try:
        keyword1 = outline.split()[-1]
        keyword2 = outline.split()[-2] + " " + outline.split()[-1]
        keyword3 = outline.split()[-3] + " " + outline.split()[-2] + " " + outline.split()[-1]
        if keyword1 in dt_limit_values or keyword2 in dt_limit_values or keyword3 in dt_limit_values:
            return float(outline.split()[0])
        else:
            return age
    except:
        return age

def scrap_age(n):
    text = "\n"
    logfiles = sorted(glob.glob("gridwork/work_*/run.log"))
    for i in range(n):
        try:
            logfile = logfiles[i]
            num = int(logfile.split("/")[-2].split("_")[-1])
        except:
            logfile = ""
            num = i
        age = 0
        if os.path.exists(logfile):
            with open(logfile, "r") as f:
                for outline in f:
                    age = process_outline(outline, age)
        if age > 0:
            if age < 1/365:
                age_str = f"[b]Age: [cyan]{age*365*24:.4f}[/cyan] hours"
            elif 1/365 < age < 1:
                age_str = f"[b]Age: [cyan]{age*365:.4f}[/cyan] days"
            elif 1 < age < 1000:
                age_str = f"[b]Age: [cyan]{age:.3f}[/cyan] years"
            else:
                age_str = f"[b]Age: [cyan]{age:.3e}[/cyan] years"
            text += f"[b][i]Track[/i] [magenta]{num}[/magenta] [yellow]----->[/yellow] {age_str}\n"
        else:
            text += f"[b][i]Track[/i] [magenta]x[/magenta] [yellow]----->[/yellow] Initiating...\n"
    return text

def progress_columns():
    '''Define progress bar columns'''
    progress_columns = (progress.SpinnerColumn(),
                progress.TextColumn("[progress.description]{task.description}"),
                progress.BarColumn(bar_width=60),
                progress.MofNCompleteColumn(),
                progress.TaskProgressColumn(),
                progress.TimeElapsedColumn())
    return progress_columns

def live_display(n):
    '''Define live display
    Args:   n (int): number of tracks
    Returns:    live_disp (rich.live.Live): live display
                progressbar (rich.progress.Progress): progress bar
                group (rich.console.Group): group of panels
    '''
    ## Progress bar
    progressbar = progress.Progress(*progress_columns(), disable=False)
    group = console.Group(panel.Panel(progressbar, expand=False), panel.Panel(scrap_age(n), expand=False))
    return live.Live(group), progressbar, group

@contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)

def create_grid_dirs(overwrite=None):
    '''
    Create grid directories. 
    Args:   overwrite (bool): overwrite existing grid directories.
                            If overwrite is None, prompt user to overwrite existing grid directories.
    '''
    ## Create archive directories
    if overwrite:
        if os.path.exists("grid_archive"):
            try:
                shutil.rmtree("grid_archive")
            except:
                try:
                    os.system("rm -rf grid_archive")
                except:
                    pass
        os.mkdir("grid_archive")
        os.mkdir("grid_archive/tracks")
        os.mkdir("grid_archive/histories")
        os.mkdir("grid_archive/profiles")
        os.mkdir("grid_archive/gyre")
        os.mkdir("grid_archive/failed")
    elif overwrite is None:
        if prompt.Confirm.ask("Overwrite existing grid_archive?"):
            shutil.rmtree("grid_archive")
    ## Create work directory
    if os.path.exists("gridwork"):
        if overwrite:
            shutil.rmtree("gridwork")
            os.mkdir("gridwork")


def archive_LOGS(name, track, save_track, gyre):
    path = os.path.abspath(os.path.join(os.getcwd().split("MESA-grid")[0], "MESA-grid"))
    os.chdir(path)
    shutil.copy(f"{name}/LOGS/history.data", f"grid_archive/histories/history_{track}.data")
    shutil.copy(f"{name}/LOGS/profiles.index", f"grid_archive/profiles/profiles_{track}.index")
    if gyre:
        gyre_archive = os.path.abspath(f"grid_archive/gyre/freqs_{track}")
        os.mkdir(gyre_archive)
        for file in glob.glob(os.path.join(name, "LOGS/*-freqs.dat")):
            shutil.copy(file, gyre_archive)
    if save_track:
        compressed_file = f"grid_archive/tracks/track_{track}.tar.gz"
        with tarfile.open(compressed_file, "w:gz") as tarhandle:
            tarhandle.add(name, arcname=os.path.basename(name))
    shutil.rmtree(name)


def read_error(name):
    retry_type = ""
    terminate_type = ""
    with open(f"{name}/run.log", "r") as f:
        for outline in f:
            splitline = outline.split(" ")
            if "retry:" in splitline:
                retry_type = outline.split(" ")
            if "terminated" in splitline and "evolution:" in splitline:
                terminate_type = outline.split(" ")
    print(retry_type, terminate_type)
    return retry_type, terminate_type


### Testing helpers

def get_data(logs_dir):
    freqs = []
    profs = []
    n_profs = []
    prof_index = pd.read_csv(f"{logs_dir}/profiles.index", skiprows=1, names=['model_number', 'priority', 'profile_number'], delim_whitespace=True)

    for f in sorted(glob.glob(f"{logs_dir}/profile*.data.GYRE"), 
                    key=lambda x: int(os.path.basename(x).split('.')[0].split('profile')[1])):
        profs.append(pd.read_table(f, skiprows=5, sep='\s+'))

    for f in sorted(glob.glob(f"{logs_dir}/profile*-freqs.dat"), 
                    key=lambda x: int(os.path.basename(x).split('.')[0].split('profile')[1].split('-')[0])):
        freqs.append(pd.read_table(f, skiprows=5, sep='\s+'))
        n_profs.append(int(f.split('profile')[-1].split('-')[0]))
    hist = pd.read_table(glob.glob(f"{logs_dir}/history.data")[0], skiprows=5, sep='\s+')
    return hist, freqs, profs, n_profs, prof_index


def fit_radial(ts, degree=0):
    """
    Fits a straight line to the radial mode frequencies. Optionally, can be used on non-radial modes.
    Only modes with radial orders 5-9 are used, as the ridges should be vertical here.
    
    Input: Theoretical (or observed) spectrum in pandas df format; mode degree to be used (default 0 = radial)
    Output: The length of the series used, and the slope, intercept, r_value, p_value, and std_err of the line.
    """
    n_min, n_max = 5, 9
    try:
        vert_freqs = ts.query("n_g == 0").query(f"l=={degree}").query(f"n_pg>={n_min}").query(f"n_pg<={n_max}")[["n_pg","Re(freq)"]].values
    except:
        vert_freqs = ts.query(f"l_obs=={degree}").query(f"n_obs>={n_min}").query(f"n_obs<={n_max}")[["n_obs","f_obs"]].values
    if len(vert_freqs>0):
        slope, intercept, r_value, p_value, std_err = linregress(vert_freqs[:,0], vert_freqs[:,1])
    else:
        slope, intercept, r_value, p_value, std_err = np.zeros(5)
    return len(vert_freqs), slope, intercept, r_value, p_value, std_err

def model_epsilon(ts):
    """
    Calls the fit_radial function to determine the epsilon value for a star's pulsations.
    
    Input: Theoretical (or observed) spectrum in pandas df format.
    Output: Epsilon
    """
    length_rad, slope, intercept, r_value, p_value, std_err = fit_radial(ts, degree=0)
    eps = intercept/slope
    if length_rad < 3:
        length_dip, slope, intercept, r_value, p_value, std_err = fit_radial(ts, degree=1)
        if length_dip > length_rad:
            eps = intercept/slope - 0.5 # take the ell=1 values and subtract 0.5 to equal epsilon (ell=0)
    return np.round(eps, 3)

def model_Dnu(ts):
    """
    Calls the fit_radial function to determine the Delta nu value for a star's pulsations.
    
    Input: Theoretical (or observed) spectrum in pandas df format.
    Output: Delta nu
    """
    length_rad, slope, intercept, r_value, p_value, std_err = fit_radial(ts, degree=0)
    if length_rad < 3:
        length_dip, slope, intercept, r_value, p_value, std_err = fit_radial(ts, degree=1)
        if length_rad > length_dip:
            # redo radial
            length_rad, slope, intercept, r_value, p_value, std_err = fit_radial(ts, degree=0)
    Dnu = slope
    return np.round(Dnu, 3)

def get_fit(l, freq):
    Dnu = model_Dnu(freq)
    epsilon = model_epsilon(freq)
    return Dnu, epsilon

def model_nlm(ts):
    """
    Creates strings that uniquely identify each mode by the mode IDs.
    
    Input: theoretical spectrum (a pandas df)
    Output: theoretical spectrum (a pandas df) with the string column added.
    """
    id_strings = []
    if "m" not in ts.columns:
        ts["m"] = np.zeros(len(ts))
    for i,row in ts.iterrows():
        id_strings.append(str(int(row["n_pg"]))+str(int(row["l"]))+str(int(row["m"])))
    ts["nlm"] = id_strings
    return ts
