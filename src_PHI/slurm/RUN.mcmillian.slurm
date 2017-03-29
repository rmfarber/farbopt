#!/bin/bash
#SBATCH -J MPI        # Job Name
#SBATCH -o MPI.o%j    # Output and error file name (%j expands to jobID)
#SBATCH --nodes=2            # Total nodes
#SBATCH -c 256
#SBATCH -p knl  # Queue (partition) name -- normal, development, etc.
#SBATCH -t 00:10:00     # Run time (hh:mm:ss)

module load intel intel-mpi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
TIME=30 #seconds
srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node=1 -c $SLURM_CPUS_PER_TASK numactl -m 1 ./mpiNloptTrain.x train.dat param.dat $TIME

