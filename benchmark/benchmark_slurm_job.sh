#!/bin/bash

#SBATCH -c 12
#SBATCH -N 1
#SBATCH -t 0-4:00
#SBATCH -p huce_intel
#SBATCH --mem=50000
#SBATCH --mail-type=END

#============================================================================
# This us a sample SLURM script that you can use to submit
# the run_1mo_benchmark.py or the run_1yr_benchmark.py
# script to a computational queue.
#
# You can modify the SLURM parameters above for your setup.
#============================================================================

# Need to initialize the bash shell so we find the conda environment
. ~/.bashrc

# Make sure to set multiple threads; Joblib will use multiple
# cores to parallelize certain plotting operations.
export OMP_NUM_THREADS=12
export OMP_STACKSIZE=500m

# Turn on Python environment (edit for your setup)
conda activate gcpy_env

# Pick the config file for the type of benchmark you are doing
config_file="1mo_benchmark.yml"
#config_file="1yr_fullchem_benchmark_yml"
#config_file="1yr_tt_benchmark.yml"

# Generate the plots
./run_benchmark.py ${config_file} > bmk.log 2>&1

# Turn off python environment
conda deactivate

exit 0

