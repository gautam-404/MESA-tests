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

############ MESA environment variables ###############
export MESASDK_ROOT=/scratch/qq01/ag9272/workspace/software/mesasdk
source $MESASDK_ROOT/bin/mesasdk_init.sh
export MESA_DIR=/scratch/qq01/ag9272/workspace/software/mesa-r23.05.1
export OMP_NUM_THREADS=2      ## max should be 2 times the cores on your machine
export GYRE_DIR=$MESA_DIR/gyre/gyre
#######################################################

source ~/.bashrc
module restore MESA
cd $PBS_O_WORKDIR

python3 params_test.py > params_test.log 2>&1

