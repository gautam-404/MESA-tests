import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

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

if __name__ == "__main__":
    archive = "../MESA-grid/grid_archive/histories/"
    tracks = pd.read_csv("../MESA-grid/track_index.dat", delim_whitespace=True)

    # selected_tracks = tracks.query("M==1.2").query("V==10")
    selected_tracks = tracks.query("V==18")
    indexes = selected_tracks["Index"]
    Z_in = selected_tracks["Z"].unique()

    fig = plt.figure(figsize=(10, 8))
    palette = sns.color_palette("viridis", n_colors=len(Z_in))
    sns.set_palette(palette)
    cmap = mpl.colors.ListedColormap(palette)
    for index in indexes:
        try:
            Teff, L = get_teff_L(get_hist(index, archive))
        except:
            pass
        plt.plot(10**Teff, L, lw=0.3)
        
    plt.ylabel(r'$\log_{10} \frac{L}{L_{\odot}}$', fontsize=25)
    plt.xlabel(r'$T_{eff}\ (K)$', fontsize=15)
    plt.xlim(4000, 10000)
    plt.grid(alpha=0.5)
    z = [[0,0],[0,0]]
    levels = Z_in
    contour = plt.contourf(z, levels, cmap=cmap)
    cb = fig.colorbar(contour, ticks=Z_in, 
                    boundaries=np.arange(len(Z_in)+1)-0.5, label=r'$Z_{in}$')
    cb.set_label(label=r'$Z_{in}$', size=15, weight='bold')
    plt.gca().invert_xaxis()
    plt.savefig("hr_diagram.png", dpi=300)
