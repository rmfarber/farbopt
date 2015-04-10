#!/bin/bash -l
#SBATCH --job-name="gpu.nlpca"
#SBATCH -o gpuMPI.o%j    # Output and error file name (%j expands to jobID)
#SBATCH --nodes=1024
#SBATCH --ntasks=1024
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:02:00

source ./common.sh

date
set +x
/usr/bin/time -p  aprun -n 1024 -N 1 -d 1 -j 1  ./mpiTrain_$APP.acc NameDoesNotMatter.dat $APP.param gpuTiming
