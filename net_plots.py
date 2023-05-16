import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from scipy.stats import linregress
import pandas as pd
import glob
import os

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times"] + plt.rcParams["font.serif"]
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['mathtext.rm'] = 'Serif'
mpl.rcParams['mathtext.it'] = 'Serif'
mpl.rcParams['mathtext.bf'] = 'Serif'
mpl.rcParams['text.usetex']= False
plt.rc('axes', linewidth=2)
plt.rc('font', weight='bold')


def get_data(logs_dir):
    freqs = []
    profs = []
    n_profs = []
    # prof_index = pd.read_table(f"{logs_dir}/profiles.index", skiprows=1, sep='\s+')
    prof_index = np.loadtxt(f"{logs_dir}/profiles.index", skiprows=1, dtype=int)

    for f in sorted(glob.glob(f"{logs_dir}/profile*.data.GYRE"), 
                    key=lambda x: int(os.path.basename(x).split('.')[0].split('profile')[1])):
        profs.append(pd.read_table(f, skiprows=5, sep='\s+'))

    for f in sorted(glob.glob(f"{logs_dir}/profile*-freqs.dat"), 
                    key=lambda x: int(os.path.basename(x).split('.')[0].split('profile')[1].split('-')[0])):
        freqs.append(pd.read_table(f, skiprows=5, sep='\s+'))
        n_profs.append(int(f.split('profile')[-1].split('-')[0]))
    hist = pd.read_table(glob.glob(f"{logs_dir}/history.data")[0], skiprows=5, sep='\s+')

    name = "/".join(logs_dir.split('/')[:-1])
    with open(f'{name}/run.log', 'r') as f:
        last_line = f.readlines()[-2]
    evo_time = float(last_line.split(' ')[-2])
    print(f"Time taken: {evo_time} s")
    return hist, freqs, profs, n_profs, prof_index, evo_time

hist, freqs, profs, n_profs, prof_index, evo_time1 = get_data("test/test_net0/LOGS")
hist, freqs, profs, n_profs, prof_index, evo_time2 = get_data("test/test_net1/LOGS")
hist, freqs, profs, n_profs, prof_index, evo_time3 = get_data("test/test_net2/LOGS")
hist, freqs, profs, n_profs, prof_index, evo_time4 = get_data("test/test_net3/LOGS")