#!/bin/bash

#SBATCH --partition=small

# set the number of nodes
#SBATCH --nodes=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=FasterRCNN_rudder

# set number of GPUs
#SBATCH --gres=gpu:1

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=jrs596@york.ac.uk


source activate Detectron2

srun python '/jmain02/home/J2AD016/jjw02/jjs00-jjw02/scripts/Elodea/Detectron2_train.py'


