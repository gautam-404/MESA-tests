import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython import display

def get_hist(index, archive):
    try:
        h = pd.read_csv(f"{archive}/history_{index}.data", skiprows=5, delim_whitespace=True)
    except:
        print(f"Missing history file {index}")
    return h

def get_teff_L(hist):
    Teff = hist["log_Teff"]
    L = hist["log_L"]
    return Teff, L

def plot_hr(v_in):
    plt.cla()
    selected_tracks = tracks.query(f"V=={v_in}")
    indexes = selected_tracks["Index"]
    Z_in = selected_tracks["Z"].unique()
    palette = sns.color_palette("viridis", n_colors=len(Z_in))
    sns.set_palette(palette)
    cmap = mpl.colors.ListedColormap(palette)
    norm = mpl.colors.Normalize(vmin=Z_in.min(), vmax=Z_in.max())
    for i, index in enumerate(indexes):
        try:
            Teff, L = get_teff_L(get_hist(index, archive))
        except:
            pass
        color = cmap(norm(selected_tracks.loc[selected_tracks['Index'] == index, 'Z'].values[0]))
        plt.plot(10**Teff, L, lw=0.3, c=color)
    plt.title(f"V = {v_in} km/s", fontsize=20)
    if v_in == 0:
        z = [[0,0],[0,0]]
        levels = Z_in
        contour = plt.contourf(z, levels, cmap=cmap)
        cb = fig.colorbar(contour, ticks=Z_in, 
                        boundaries=np.arange(len(Z_in)+1)-0.5, label=r'$Z_{in}$')
        cb.set_label(label=r'$Z_{in}$', size=15, weight='bold')
    plt.grid(alpha=0.5)
    plt.xlim(4000, 14500)
    plt.ylim(-0.5, 2)
    plt.gca().invert_xaxis()
    plt.xlabel(r'$T_{eff}\ (K)$', fontsize=15)
    plt.ylabel(r'$\log_{10} \frac{L}{L_\odot}$', fontsize=15)
    plt.draw()


if __name__ == "__main__":
    archive = "../MESA-grid/grid_archive/histories/"
    tracks = pd.read_csv("../MESA-grid/track_index.dat", delim_whitespace=True)

    fig = plt.figure(figsize=(10, 8))
    V = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    for v_in in V:
        plot_hr(v_in)
        plt.savefig(f"./figures/hr_diagram_{v_in}.png", dpi=300)


