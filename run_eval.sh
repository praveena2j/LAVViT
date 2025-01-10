#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=75G
#SBATCH --job-name=ViTforSV
#SBATCH --time=07-00:00:00
#SBATCH --mail-user=gnana-praveen.rajasekhar@crim.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out
export PATH="/misc/scratch11/anaconda3/bin:$PATH"
#source activate py36_ssd20
source activate LAVISH
python main.py \
--Adapter_downsample 8 \
--train_audio_folder /misc/scratch02/reco/Corpora/VoxCeleb_August2021/voxceleb1 \
--batch_size 2 \
--decay 0.35 \
--decay_epoch 3 \
--early_stop 5 \
--epochs 50 \
--is_audio_adapter_p1 1 \
--is_audio_adapter_p2 1 \
--is_audio_adapter_p3 0 \
--is_before_layernorm 1 \
--is_bn 1 \
--is_fusion_before 1 \
--is_gate 1  \
--is_post_layernorm 1 \
--is_vit_ln 0 \
--lr 1e-04 \
--lr_mlp 5e-06 \
--lr_loss 0.0001 \
--mode Eval \
--model MMIL_Net \
--num_conv_group 2 \
--num_tokens 2 \
--num_workers 16 \
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