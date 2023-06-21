#!/bin/bash -l
 
#PBS -N MESA_test
#PBS -P 0201
#PBS -q default
#PBS -l select=1:ncpus=128:mem=128GB
#PBS -l walltime=24:00:00
#PBS -o net.out
#PBS -j oe 
##PBS -M anuj.gautam@usq.edu.au
##PBS -m abe

cd $PBS_O_WORKDIR

python test.py > test.log 2>&1

