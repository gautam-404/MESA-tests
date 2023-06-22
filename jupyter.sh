#!/bin/bash -l

#PBS -N jupyter
#PBS -P 0201
#PBS -q default
#PBS -l select=1:ncpus=32:mem=128GB
#PBS -l walltime=12:00:00
#PBS -o jupyter.log
#PBS -j oe 

source ~/.bashrc
cd $PBS_O_WORKDIR

jupyter notebook --no-browser --NotebookApp.allow_origin='*' --port=8889 --ip=0.0.0.0 > jupyter.log 2>&1
