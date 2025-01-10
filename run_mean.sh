#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=98G
#SBATCH --job-name=ViTforSV
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=gnana-praveen.rajasekhar@crim.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out
export PATH="/misc/scratch11/anaconda3/bin:$PATH"
#source activate py36_ssd20
source activate LAVISH
python main_mean.py \
--train_audio_folder /misc/scratch02/reco/Corpora/VoxCeleb_August2021/voxceleb1 \
--train_video_folder /misc/scratch11/Voxceleb1_Data/Voxceleb1_Data/Insightface_Aligned_Images/ \
--vis_encoder_type vit \
--train_list /misc/lu/fast_scratch/patx/rajasegp/AV_Cleanse_withRJCA/AVCleanse-main/face/train_list_face_new.txt \
--train_path /misc/scratch11/Voxceleb1_Data/Voxceleb1_Data/Insightface_Aligned_Images/ \
--eval_trials /misc/lu/fast_scratch/patx/rajasegp/AV_Cleanse_withRJCA/veri_val_face.txt \
--eval_list /misc/lu/fast_scratch/patx/rajasegp/AV_Cleanse_withRJCA/AVCleanse-main/face/val_list_face_new.txt \
--eval_path /misc/scratch11/Voxceleb1_Data/Voxceleb1_Data/Insightface_Aligned_Images/ \
--save_path exps/debug \
--n_class 1150 \
--scale_a 64 \
--margin_a 0.4 \