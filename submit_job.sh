#!/bin/bash
#SBATCH --account=sta440-f20
#SBATCH -p common
#SBATCH -N1
#SBATCH -c1
#SBATCH --mem=8G

module load Python/3.8.1
python load_wesad.py
