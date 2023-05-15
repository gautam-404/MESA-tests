#!/bin/bash -l
 
#PBS -N MESA+GYRE
#PBS -P 0201
#PBS -q default
#PBS -l select=1:ncpus=128:mem=64GB
#PBS -l walltime=12:00:00
#PBS -o joblogs/job.out
#PBS -j oe 
##PBS -M anuj.gautam@usq.edu.au
##PBS -m abe


cd $PBS_O_WORKDIR

# python test_profiles.py > ../joblogs/output.log 2>&1
python test.py > ../joblogs/output.log 2>&1

# python script.py -t mg -s incomplete_tracks.dat > joblogs/output_gyre.log 2>&1
# cp -r grid_archive_gadi/histories grid_archive_gadi/profile_indexes grid_archive/
# python script.py -t mg --ray > joblogs/output.log 2>&1

