#!/bin/bash
#SBATCH -J MPI        # Job Name
#SBATCH -o MPI.o%j    # Output and error file name (%j expands to jobID)
#SBATCH --nodes=1            # Total nodes
#SBATCH -p knl  # Queue (partition) name -- normal, development, etc.
#SBATCH -t 00:10:00     # Run time (hh:mm:ss)

module load intel intel-mpi

SIZE=60000001
./genData.x - $SIZE 0.01 123 > train.dat

