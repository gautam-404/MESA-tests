import tarfile as tar
import glob
from rich import print, progress

gyre_tars = glob.glob("/home/u1159830/workspace/MESA-grid/grid_urot/gyre/*.tar.gz")

with progress.Progress() as progress:
    task = progress.add_task("[green]Extracting Gyre models...", total=len(gyre_tars))
    for t in gyre_tars:
        with tar.open(t) as f:
            f.extractall("/home/u1159830/workspace/MESA-grid/grid_urot/gyre/")
            progress.advance(task)

