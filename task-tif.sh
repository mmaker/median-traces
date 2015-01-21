#!/bin/bash
#PBS -V
#PBS -N median-traces-unc
#PBS -l walltime=6666:00:00
#PBS -m bea
#PBS -M michele@tumbolandia.net
cd /home/michele/median-traces

./generate_dataset ucid.v2.tar.gz 101
./learn dataset-j101/ || exit 1
./test  dataset-j101/ || exit 2

exit 0
