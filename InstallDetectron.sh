#!/bin/sh

##Job Script for FYP

#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --job-name=FYPJob
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda
module load gcc/10.4.0
source activate FinalEnv
python -m pip install -e detectron2