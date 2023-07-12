import glob
import numpy as np
import os
import pandas as pd
# import scipy as sp
from scipy import stats
# from scipy import interpolate
from tqdm import tqdm
import multiprocessing as mp
import copy


# functions to read profile indexes
def import_histories(args):
    hfn, pfn, m_i, z_i, v_i, tr_num = args
    h = pd.read_csv(hfn, delim_whitespace=True, skiprows=5)
    p = pd.read_csv(pfn, skiprows=1, names=['model_number', 'priority', 'profile_number'], delim_whitespace=True)
    h = pd.merge(h, p, on='model_number', how='outer')
    h["Myr"] = h["star_age"] * 1.0E-6
    h["m"] = m_i
    h["z"] = z_i
    h["v"] = v_i
    h["tr_num"] = tr_num
    h["density"] = h["star_mass"] / np.power(10, h["log_R"]) ** 3
    return h


def read_gyre(fn):
    ts = pd.read_csv(fn, skiprows=5, delim_whitespace=True)
    ts.drop(columns=["M_star", "R_star", "Im(freq)", "E_norm"], inplace=True)
    ts.rename(columns={"Re(freq)": "freq"}, inplace=True)
    return ts.query("n_g == 0")


def model_nlm(ts):
    id_strings = [str(int(row["n_pg"])) + str(int(row["l"])) + str(int(row["m"])) for _, row in ts.iterrows()]
    ts["nlm"] = id_strings
    return ts


def fit_radial(ts, degree=0):
    n_min, n_max = 5, 9
    try:
        vert_freqs = ts.query("n_g == 0").query(f"l=={degree}").query(f"n_pg>={n_min}").query(f"n_pg<={n_max}")[
            ["n_pg", "freq"]].values
    except:
        vert_freqs = ts.query(f"l_obs=={degree}").query(f"n_obs>={n_min}").query(f"n_obs<={n_max}")[
            ["n_obs", "f_obs"]].values
    if len(vert_freqs > 0):
        slope, intercept, r_value, p_value, std_err = stats.linregress(vert_freqs[:, 0], vert_freqs[:, 1])
    else:
        slope, intercept, r_value, p_value, std_err = np.zeros(5)
    return len(vert_freqs), slope, intercept, r_value, p_value, std_err


def epsilon(ts):
    length_rad, slope, intercept, r_value, p_value, std_err = fit_radial(ts, degree=0)
    eps = intercept / slope
    if length_rad < 3:
        length_dip, slope, intercept, r_value, p_value, std_err = fit_radial(ts, degree=1)
        if length_dip > length_rad:
            eps = intercept / slope - 0.5
    return np.round(eps, 3)


def model_Dnu(ts):
    length_rad, slope, intercept, r_value, p_value, std_err = fit_radial(ts, degree=0)
    if length_rad < 3:
        length_dip, slope, intercept, r_value, p_value, std_err = fit_radial(ts, degree=1)
        if length_rad > length_dip:
            length_rad, slope, intercept, r_value, p_value, std_err = fit_radial(ts, degree=0)
    Dnu = slope
    return np.round(Dnu, 3)


def process_track(track):
    tr_num, m_i, z_i, v_i = track
    hfn = f"{log_dirs[tr_num]}/history.data"
    pfn = f"{log_dirs[tr_num]}/profiles.index"
    return import_histories((hfn, pfn, m_i, z_i, v_i, tr_num))

def process_gyre_file(grid_iter):
    i, row = grid_iter
    # dino = copy.copy(row)
    dino = row
    try:
        if os.path.isfile(row.gyre_fn):
            dino["missing_gyre_flag"] = 0
            ts = read_gyre(row.gyre_fn)
            ts = model_nlm(ts)
            dino["Dnu"] = model_Dnu(ts)
            dino["eps"] = epsilon(ts)
            for j, s in enumerate(mode_strings):
                try:
                    dino[mode_labels[j]] = np.round(ts.query(f"nlm=='{s}'")["freq"].values[0], 5)
                except:
                    dino[mode_labels[j]] = np.nan
        else:
            dino["missing_gyre_flag"] = 1
    except:
        dino["missing_gyre_flag"] = 1
    
    return dino

log_dirs = sorted(glob.glob("test/r23.05.1_m1.7_z0.015_v0_net*/LOGS"))
mode_labels = ["n1ell0m0","n2ell0m0","n3ell0m0","n4ell0m0","n5ell0m0","n6ell0m0","n7ell0m0","n8ell0m0","n9ell0m0","n10ell0m0",
            "n1ell1mm1","n2ell1mm1","n3ell1mm1","n4ell1mm1","n5ell1mm1","n6ell1mm1","n7ell1mm1","n8ell1mm1","n9ell1mm1","n10ell1mm1",
            "n1ell1m0","n2ell1m0","n3ell1m0","n4ell1m0","n5ell1m0","n6ell1m0","n7ell1m0","n8ell1m0","n9ell1m0","n10ell1m0",
            "n1ell1mp1","n2ell1mp1","n3ell1mp1","n4ell1mp1","n5ell1mp1","n6ell1mp1","n7ell1mp1","n8ell1mp1","n9ell1mp1","n10ell1mp1"]
mode_strings = ["100","200","300","400","500","600","700","800","900","1000",
        "11-1","21-1","31-1","41-1","51-1","61-1","71-1","81-1","91-1","101-1",
        "110","210","310","410","510","610","710","810","910","1010",
        "11-1","21-1","31-1","41-1","51-1","61-1","71-1","81-1","91-1","101-1"]
if __name__ == "__main__":
    log_dirs = sorted(glob.glob("test/r23.05.1_m1.7_z0.015_v0_net*/LOGS"))
    m = [float(name.split("/")[1].split("_")[1].split("m")[1]) for name in log_dirs]
    z = [float(name.split("/")[1].split("_")[2].split("z")[1]) for name in log_dirs]
    v = [float(name.split("/")[1].split("_")[3].split("v")[1]) for name in log_dirs]
    num = [i for i in range(len(log_dirs))]
    tracks = pd.DataFrame(np.array([num, m, z, v]).T, columns=["tr_num", 'm', 'z', 'v'])
    tracks = tracks.astype({'tr_num':int, 'm': float, 'z': float, 'v': float})
    tracks.sort_values(by='m', inplace=True)
    tracks.reset_index(drop=True, inplace=True)

    # Process the tracks
    print("Importing histories and creating dataframe...")

    track_histories = [log_dir + "/history.data" for log_dir in log_dirs]

    # Use multiprocessing to parallelize track processing
    with mp.Pool() as pool:
        track_args = [(int(track.tr_num), track.m, track.z, track.v) for _, track in tracks.iterrows()]
        track_histories = list(tqdm(pool.imap(process_track, track_args), desc="Processing tracks", total=len(tracks)))

    grid = pd.concat(track_histories)

    grid["teff"] = np.round(np.power(10,grid["log_Teff"]),2)
    grid.rename(columns={"log_L":"logL","delta_nu":"dnu_muhz"},inplace=True)
    # drop_cols = ['model_number','num_zones','star_age','log_dt','he_core_mass', 
    #             'center_h1', 'center_he4', 'log_LH', 'log_LHe', 'log_LZ', 'log_Lnuc', 'pp', 'cno', 
    #             'tri_alpha', 'log_Teff', 'average_h1', 'average_he4', 'total_mass_h1', 'total_mass_he4', 
    #             'log_cntr_P', 'log_cntr_Rho', 'log_cntr_T', 'num_retries','num_iters','priority']
    # grid.drop(columns=drop_cols,inplace=True)

    grid = grid.query("profile_number==profile_number")
    grid = grid.astype({"profile_number": int, "tr_num": int})

    # generate all of the Gyre filenames
    print("Generating Gyre filenames...")
    gyre_fns = []
    for i,row in tqdm(grid.iterrows()):
        v = row["v"]
        p = int(row["profile_number"])
        tr_num = int(row["tr_num"])
        try:
            gyre_fns.append(log_dirs[tr_num] + f"/profile{p}-freqs.dat")
        except:
            gyre_fns.append(" ")
    grid["gyre_fn"] = gyre_fns


    # prepare to read and assign modes from gyre files to df
    print("Reading and assigning modes...")
    for s in mode_labels:
        grid[s] = np.repeat(np.nan, len(grid))
    grid["missing_gyre_flag"] = np.repeat(np.nan, len(grid))
    grid["Dnu"] = np.repeat(np.nan, len(grid))
    grid["eps"] = np.repeat(np.nan, len(grid))

    # Use multiprocessing to parallelize gyre file processing
    with mp.Pool() as pool:
        grid = list(tqdm(pool.imap(process_gyre_file, grid.iterrows()), desc=f"Processing gyre files...", total=len(grid)))

    megasaurus = pd.DataFrame(grid)
    # megasaurus.to_csv("megasaurus_parallel.csv", index=False)

    # Drop rows where there was no gyre data, hence no successful Dnu calculation
    mini = megasaurus.query("Dnu==Dnu")

    l = 1
    for n in range(1, 11):
        # mini[f"n{n}ell{l}dfreq"] = mini[f"n{n}ell{l}m0"] - mini[f"n{n}ell{l}mm1"]
        mini = mini.assign(**{f"n{n}ell{l}dfreq": mini[f"n{n}ell{l}m0"] - mini[f"n{n}ell{l}mm1"]})
        mini.drop(columns=[f"n{n}ell{l}mp1", f"n{n}ell{l}mm1"], inplace=True)

    # Drop columns that we don't need for the neural network
    mini.drop(columns=['nu_max', 'missing_gyre_flag'], inplace=True)

    mini.to_csv("minisaurus_parallel.csv", index=False)
