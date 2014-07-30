#!/bin/bash
source ./common.sh
#!/bin/bash
#SBATCH -J offloadMPI        # Job Name
#SBATCH -o offloadMPI.o%j    # Output and error file name (%j expands to jobID)
#SBATCH -N 2            # Total nodes
#SBATCH -n 2           # Total number of mpi tasks requested
#SBATCH -p normal-mic  # Queue (partition) name -- normal, development, etc.
#SBATCH -t 00:10:00     # Run time (hh:mm:ss) - 1.5 hours
#./gen_$APP foo.dat $EXAMPLES $VAR 1234 
ibrun ./mpiTrain_$APP.off foo.dat $APP.param mpiTiming
