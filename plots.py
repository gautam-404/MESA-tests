import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from scipy.stats import linregress
import pandas as pd
from ipywidgets import interact, IntSlider
import glob
import os
import scienceplots
import imageio

plt.style.use('science')
mpl.rcParams.update(mpl.rcParamsDefault)

def get_data(logs_dir):
    logs = sorted(glob.glob(f"{logs_dir}/V*"), key=lambda x: float(x.split('/')[-1].split('_')[-1]))
    v = [float(log.split('/')[-1].split('_')[-1]) for log in logs]
    freqs = []
    hists = []
    profs = []
    for log in logs:
        freqs.append(pd.read_table(glob.glob(f"{log}/profile*-freqs.dat")[0], skiprows=5, sep='\s+'))
        hists.append(pd.read_table(glob.glob(f"{log}/history.data")[0], skiprows=5, sep='\s+'))
        profs.append(pd.read_table(glob.glob(f"{log}/profile*.data.GYRE")[0], skiprows=5, sep='\s+'))
    return v, freqs, hists, profs


def hist_plots(logs_dir, col, unit='', v_selected=[], logarithmic=True):
    v, freqs, hists, profs = get_data(logs_dir)
    plt.figure(figsize=(10, 6))
    if v_selected == []:
        v_selected = v
    with sns.color_palette("flare", n_colors=len(v_selected)) as palette:
        cmap = mpl.colors.ListedColormap(palette)
        for v_i in range(len(v)):
            if v[v_i] in v_selected:
                t = hists[v_i]['star_age']/1e6
                if logarithmic:
                    plt.plot(t, 10**hists[v_i][col], label=f"V_in = {v[v_i]} {unit}")
                else:
                    plt.plot(t, hists[v_i][col], label=f"V_in = {v[v_i]} {unit}")
        
    plt.xlabel('Age (Myrs)')
    if unit != '':
        unit = f"  ({unit})"
    if logarithmic:
        plt.ylabel(' '.join(col.split("_")[1:])+unit)
    else:
        plt.ylabel(col+unit)
    Z = [[0,0],[0,0]]
    levels = range(0, 22, 2)
    contour = plt.contourf(Z, levels, cmap=cmap)
    plt.colorbar(contour, ticks=v, boundaries=np.arange(len(v_selected)+1)-0.5, label='V_in (km/s)')
    if "steps" in logs_dir.split('_')[-1]:
        M = logs_dir.split('_')[-3].split('M')[-1]
        Z = logs_dir.split('_')[-2].split('Z')[-1]
        steps = logs_dir.split('_')[-1].split('steps')[-1]
        plt.title(f"M = {M}, Z = {Z}, Steps = {steps}")
    else:
        M = logs_dir.split('_')[-2].split('M')[-1]
        Z = logs_dir.split('_')[-1].split('Z')[-1]
        plt.title(f"M = {M}, Z = {Z}")
    # plt.legend()
    # plt.show()


def plot_freq(logs_dir, n, l):
    v, freqs, hists, profs = get_data(logs_dir)
    fig, ax = plt.subplots()
    for i, v_i in enumerate(v):
        freq = freqs[i]
        f_i = freq[np.logical_and(freq.l == l, freq.n_p == n)]
        f = f_i['Re(freq)'].values
        m = f_i['m'].values
        vv = np.ones_like(f) * v_i
        scatter = ax.scatter(vv, f, label=f"m = {m}", c=m);

    # legend1 = ax.legend(*scatter.legend_elements(),
    #                     loc="lower left", title="m")
    # ax.add_artist(legend1)
    plt.xticks(ticks=v)
    plt.xlabel(r'$V_{in}$ (km/s)')
    plt.ylabel(f'Frequency (n = {n}, l = {l}), $d^{-1}$')
    # plt.show()
    plt.savefig(f"tests_here/test_profiles/figures/freqs_n_{n}_l_{l}.png", dpi=300)

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

def get_fit(freq):
    Dnu = model_Dnu(freq)
    epsilon = model_epsilon(freq)
    return Dnu, epsilon


##ECHELLE
def plot_echelle(logs_dir, v_i):
    v, freqs, hists, profs = get_data(logs_dir)
    fig, ax = plt.subplots()
    hist = hists[v_i]
    prof = profs[v_i]
    freq = freqs[v_i]

    nu_max = hist.nu_max.values[-1]
    freq = freq[freq['Re(freq)'] < 5/3*nu_max]
    colors = ('black', '#8B3A3A')
    colors2 = ('#EE6363', '#8B3A3A', '#EE6363')
    Dnu, epsilon = get_fit(freq)
    for ell in np.unique(freq.l.values):
        nus = freq[freq.l == ell]
        if ell == 0:
            x = np.append(nus['Re(freq)']%Dnu, nus['Re(freq)']%Dnu + Dnu)
            y = np.append(nus['Re(freq)'], nus['Re(freq)'])
            plt.plot(x, y, 'o', 
                    mfc=colors[ell], mec=colors[ell], alpha=0.85,
                    ms=6, mew=1, 
                    label=str(ell))
        else:
            nus = nus[np.logical_and(nus.n_p >= 1, nus.n_p <= 9)]
            nu_m0 = nus[nus.m == 0]['Re(freq)'].values
            x = np.append(nu_m0%Dnu, nu_m0%Dnu + Dnu)
            y = np.append(nu_m0, nu_m0)
            plt.plot(x, y, 'o',
                    mfc=colors2[1], mec=colors2[1], alpha=0.85,
                    ms=6, mew=1,
                    label=str(ell))

            nu_m1 = nus[nus.m == 1]['Re(freq)'].values
            x = np.append(nu_m1%Dnu, nu_m1%Dnu + Dnu)
            y = np.append(nu_m1, nu_m1)
            plt.plot(x, y, '+',
                    mfc=colors2[2], mec=colors2[2], alpha=0.85,
                    ms=12, mew=1)
            plt.plot(x, y, 'o',
                    mfc=colors2[2], mec=colors2[2], alpha=0.85,
                    ms=3, mew=1)
            
            nu_m_1 = nus[nus.m == -1]['Re(freq)'].values
            x = np.append(nu_m_1%Dnu, nu_m_1%Dnu + Dnu)
            y = np.append(nu_m_1, nu_m_1)
            plt.plot(x, y, '_',
                    mfc=colors2[0], mec=colors2[0], alpha=0.85,
                    ms=12, mew=1)
            plt.plot(x, y, 'o',
                    mfc=colors2[0], mec=colors2[0], alpha=0.85,
                    ms=3, mew=1)
    plt.legend(title='$\u2113$', loc='upper right')

    plt.axvline(8.2, ls='--', c='darkgray', zorder=-99)

    plt.ylim([0, 140])
    plt.xlim([0, 11])

    plt.ylabel(r'Frequency, $d^{-1}$')
    plt.xlabel(r'$\nu\; \rm{mod}\; \Delta\nu/d^{-1}$')
    plt.title(f'V={v[v_i]} km/s', size=20)



if __name__ == '__main__':
    # hist_plots('tests_here/test_profiles_M1.4_Z0.008_steps', 'surf_avg_v_rot', unit="km/s", logarithmic=False)
    # plt.savefig('surf_avg_v_rot.png', dpi=300)


    # folders = glob.glob('tests_here/test_profiles_M1.4_Z*')
    # i = 0
    # for folder in folders: 
    #     hist_plots(folder, 'surf_avg_v_rot', unit="km/s", logarithmic=False)
    #     plt.savefig(f'surf_avg_v_rot_{i}.png', dpi=300)
    #     i += 1

    
    logs_dirs = glob.glob('tests_here/test_profiles_M1.4_Z*')
    for logs_dir in logs_dirs:
        if not os.path.exists(f"{logs_dir}/echelle"):
            os.mkdir(f"{logs_dir}/echelle")
        v, freqs, hists, profs = get_data(logs_dir)
        for i in range(len(v)):
            plot_echelle(logs_dir, i)
            plt.savefig(f"{logs_dir}/echelle/echelle_v_{v[i]}.png", dpi=300)
            plt.close()


    logs_dirs = glob.glob('tests_here/test_profiles_M1.4_Z*')
    for logs_dir in logs_dirs:
        steps = logs_dir.split('_')[-1].split('steps')[-1]
        print(steps)
        v_list = np.arange(0, 22., 2)
        frames = []
        for v in v_list:
            image = imageio.v2.imread(f"{logs_dir}/echelle/echelle_v_{v}.png")
            frames.append(image)
        imageio.mimsave(f'./echelle_steps{steps}.gif', # output gif
                    frames,          # array of input frames
                    duration = 0.3)
                    # fps = 5)         # optional: frames per second