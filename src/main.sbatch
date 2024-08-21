#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --time=0-23:00:00
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --job-name=main
#SBATCH --mem=50G
#SBATCH --output=../logs/main.out
#SBATCH --error=../logs/main.err


module load python/anaconda-2022.05
# python figure_1.py
python figure_2.py
# python figure_3.py
# python figure_4.py