#!/bin/bash -l
 
#PBS -N MESA_test
#PBS -q express
#PBS -l ncpus=96
#PBS -l mem=100GB
#PBS -l walltime=05:00:00
#PBS -o job_gadi.out
#PBS -j oe 
##PBS -M anuj.gautam@usq.edu.au
##PBS -m abe

cd $PBS_O_WORKDIR

python params_test.py > params_test.log 2>&1

