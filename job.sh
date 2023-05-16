#!/bin/bash -l
 
#PBS -N net_tests
#PBS -P 0201
#PBS -q default
#PBS -l select=1:ncpus=128:mem=64GB
#PBS -l walltime=12:00:00
#PBS -o net.out
#PBS -j oe 
##PBS -M anuj.gautam@usq.edu.au
##PBS -m abe

cd $PBS_O_WORKDIR

python net_tests.py > net_tests.log 2>&1

