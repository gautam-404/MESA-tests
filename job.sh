#!/bin/bash -l
 
#PBS -N MESA_test
#PBS -P 0201
#PBS -q default
#PBS -l select=1:ncpus=128:mem=128GB
#PBS -l walltime=12:00:00
#PBS -o net.out
#PBS -j oe 
##PBS -M anuj.gautam@usq.edu.au
##PBS -m abe


############ MESA environment variables ###############
export MESASDK_ROOT=/home/u1159830/software/mesasdk
source $MESASDK_ROOT/bin/mesasdk_init.sh
export MESA_DIR=/home/u1159830/software/mesa-r23.05.1
export OMP_NUM_THREADS=2      ## max should be 2 times the cores on your machine
export GYRE_DIR=$MESA_DIR/gyre/gyre
#######################################################

cd $PBS_O_WORKDIR

python params_test.py > params_test.log 2>&1

