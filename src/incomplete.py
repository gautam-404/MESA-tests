import numpy as np
from rich import print
import glob
import argparse

from .grid import init_grid

parser = argparse.ArgumentParser(description="Find incomplete tracks",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dir", action="store", 
                        help="directory to search for incomplete tracks")

args = parser.parse_args()
config = vars(args)


def mesa_missing(directory):
    track_indexes_all = range(1, len(init_grid()[0])+1)
    tracks = glob.glob(f"{directory}/tracks/*")
    track_indexes = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in tracks]
    
    missing = []
    for index in track_indexes_all:
        if index not in track_indexes:
            missing.append(index)
    return missing

def gyre_missing(directory):
    tracks = glob.glob(f"{directory}/tracks/*")
    gyre_done = glob.glob(f"{directory}/gyre/*")
    gyre_indexes = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in gyre_done] 
    track_indexes = [int(name.split("/")[-1].split(".")[0].split("_")[-1]) for name in tracks]
    print(len(gyre_indexes))
    print(len(track_indexes))
    missing = []
    for t_index in track_indexes:
        if t_index not in gyre_indexes:
            missing.append(t_index)
    return missing



if __name__ == "__main__":
    if config["dir"] is None:
        raise ValueError("Please provide a directory")
    directory = config["dir"]
    # missing = gyre_missing(directory)
    missing = mesa_missing(directory)
    print(len(missing))

    np.savetxt("incomplete_tracks.dat", np.array(missing), fmt="%i")


