#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:00:00
#SBATCH --job-name=en_synthesiser
#SBATCH --mem=5G
#SBATCH --array=0
#SBATCH --output=synth_output_%a.out
#SBATCH --error=synth_errors_%a.err

module load anaconda
module load cuda 
source activate tts

srun python -u /scratch/work/lunt1/TTS/generate.py \
--partial_model_path=output_fold_ \
