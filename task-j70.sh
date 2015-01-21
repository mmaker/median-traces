#!/bin/bash
#PBS -V
#PBS -N median-traces-jpeg70
#PBS -l walltime=6666:00:00
#PBS -m bea
#PBS -M michele@tumbolandia.net
cd /home/michele/median-traces

export NOISE_RANGES=("5 9" "7 11")

NOISE_RANGES=$NOISE_RANGES ./generate_dataset ucid.v2.tar.gz 70
./learn dataset-j70/ || exit 1
NOISE_RANGES=$NOISE_RANGES ./test  dataset-j70/ || exit 2

exit 0
