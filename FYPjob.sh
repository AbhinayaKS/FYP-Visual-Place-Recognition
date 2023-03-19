#!/bin/sh

##Job Script for FYP

#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --job-name=FYPJob
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
#SBATCH --nodelist=SCSEGPU-TC1-02

module load anaconda
##module load gcc/10.4.0
source activate FinalEnv
##python main.py --mode=test --split=val --resume=vgg16_netvlad_checkpoint/
python main.py --mode=train --arch=vgg16 --pooling=netvlad --num_clusters=64 --includeSemantic=True --nEpochs=5 --cacheRefreshRate=0
##python main.py --mode=cluster --arch=vgg16 --pooling=netvlad --num_clusters=64
##python main.py --mode=label --arch=vgg16 --pooling=netvlad --num_clusters=64