import pandas as pd
import glob
import os, psutil
import numpy as np
import itertools
from scipy.stats import linregress
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
import argparse

import multiprocessing as mp
from rich import print, progress, prompt
import pickle


def get_data(index, grid_dir):
    prof_index_i = np.loadtxt(f"{grid_dir}/profiles/profiles_{index}.index", skiprows=1, dtype=int)
    freqs_i = []
    n_profs_i = []
    freq_files = sorted(glob.glob(f"{grid_dir}/gyre/freqs_track_{index}/profile*-freqs.dat"), key=lambda x: int(os.path.basename(x).split('.')[0].split('profile')[1].split('-')[0]))
    for file in freq_files:
        freqs_i.append(pd.read_table(file, skiprows=5, sep='\s+'))
        n_profs_i.append(int(file.split('profile')[-1].split('-')[0]))
    hist_i = pd.read_table(f"{grid_dir}/histories/history_{index}.data", skiprows=5, sep='\s+')
    return hist_i, freqs_i, n_profs_i, prof_index_i


def get_all(grid_dir="../MESA-grid/grid_urot"):
    hists = []
    freqs = []
    n_profs = []
    prof_indexes = []

    with progress.Progress() as pbar:
        task = pbar.add_task("[green]Reading data...", total=14872)
        for i in range(1, 14873):
            try:
                hist_i, freqs_i, n_profs_i, prof_index_i = get_data(i, grid_dir=grid_dir)
                hists.append(hist_i)
                freqs.append(freqs_i)
                n_profs.append(n_profs_i)
                prof_indexes.append(prof_index_i)
            except:
                print(f"Failed to get data for grid index {i}")
            pbar.advance(task)
    return hists, freqs, n_profs, prof_indexes


def get_data_parallel(args):
    index, grid_dir, hists, freqs, n_profs, prof_indexes = args
    try:
        hist_i, freqs_i, n_profs_i, prof_index_i = get_data(index, grid_dir=grid_dir)
        hists.append(hist_i)
        freqs.append(freqs_i)
        n_profs.append(n_profs_i)
        prof_indexes.append(prof_index_i)
    except:
        print(f"Failed to get data for grid index {index}")
        
        
def get_all_parallel(grid_dir="../MESA-grid/grid_urot"):
    manager = mp.Manager()
    hists = manager.list()
    freqs = manager.list()
    n_profs = manager.list()
    prof_indexes = manager.list()
    
    with progress.Progress() as pbar:
        task = pbar.add_task("[green]Reading data...", total=14872)
        with mp.Pool(psutil.cpu_count(logical=False)) as pool:
            args = [(i, grid_dir, hists, freqs, n_profs, prof_indexes) for i in range(1, 14873)]
            for res in pool.imap_unordered(get_data_parallel, args):
                pbar.advance(task)
    return list(hists), list(freqs), list(n_profs), list(prof_indexes)



if __name__ == '__main__':
    if prompt.ask("read or write?", choices=["read", "write"]) == "write":  
        hists_all, freqs_all, n_profs_all, prof_indexes_all = get_all_parallel("../MESA-grid/grid_urot")

        if os.path.exists('data/hists.pkl'):
            os.remove('data/hists.pkl')
        with open('data/hists.pkl', 'wb') as f:
            pickle.dump(hists_all[:], f, protocol = None, fix_imports = True)

        if os.path.exists('data/freqs.pkl'):
            os.remove('data/freqs.pkl')
        with open('data/freqs.pkl', 'wb') as f:
            pickle.dump(freqs_all[:], f, protocol = None, fix_imports = True)

        if os.path.exists('data/n_profs.pkl'):
            os.remove('data/n_profs.pkl')
        with open('data/n_profs.pkl', 'wb') as f:
            pickle.dump(n_profs_all[:], f, protocol = None, fix_imports = True)

        if os.path.exists('data/prof_indexes.pkl'):
            os.remove('data/prof_indexes.pkl')
        with open('data/prof_indexes.pkl', 'wb') as f:
            pickle.dump(prof_indexes_all[:], f, protocol = None, fix_imports = True)
    else:
        with open('data/hists.pkl', 'rb') as f:
            hists_all = pickle.load(f)
        with open('data/freqs.pkl', 'rb') as f:
            freqs_all = pickle.load(f)
        with open('data/n_profs.pkl', 'rb') as f:
            n_profs_all = pickle.load(f)
        with open('data/prof_indexes.pkl', 'rb') as f:
            prof_indexes_all = pickle.load(f)
        
    
