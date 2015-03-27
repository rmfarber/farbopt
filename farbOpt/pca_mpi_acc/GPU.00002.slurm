#!/bin/bash -l
#SBATCH --job-name="gpu.pca"
#SBATCH -o gpuMPI.o%j    # Output and error file name (%j expands to jobID)
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:01:00

source ./common.sh

date
set +x
/usr/bin/time -p  aprun -n 2 -N 1 -d 1 -j 1  ./mpiTrain_$APP.acc NameDoesNotMatter.dat $APP.param gpuTiming
