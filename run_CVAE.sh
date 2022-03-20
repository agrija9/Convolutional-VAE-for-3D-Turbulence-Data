#!/bin/bash
#
#SBATCH --job-name=ice_detection_vrae
#SBATCH -o ./output_jobs/slurm.%x_10_factor_1_epochs_64_batch_2020-06-03.%j.out # STDOUT
#SBATCH -e ./output_jobs/slurm.%x_10_factor_1_epochs_64_batch_2020-06-03.%j.err #STDERR
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=20:00:00

# insert variables with sed before calling
date=2020-20-05_16:40

echo DATE and TIME $date

# set hyperparameters
batch_size=128
epochs=100
latent_length=32

# set model name and create the output directory
model_name=cvae_epochs_${epochs}_batch_size_${batch_size}_latent_size_${latent_length}_${date}
model_name_base=cvae
output=${model_name}_$date
mkdir -p /home/apreciadogrijalva/alan_cluster/Air_ducts/output_jobs/$model_name_base

# load module (custom anaconda environment)
module load Anaconda3/5.1.0
source /home/apreciadogrijalva/.bashrc
conda activate /home/apreciadogrijalva/alan-env
echo Modules loaded....

# print model/train info
echo Model $model_name
echo Reference $output
echo Training model for $epochs iterations
echo Calculate on $cores Cores

# run python script
CUDA_VISIBLE_DEVICES=6,7 /home/apreciadogrijalva/alan-env/bin/python /home/apreciadogrijalva/alan_cluster/Air_ducts/main.py --batch_size $batch_size --epochs $epochs --z_dim $latent_length
