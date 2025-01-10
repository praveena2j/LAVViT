import os
import sys

train_path = '/misc/lu/bf_scratch/patx/rajasegp/Voxceleb2_Data'

train_list = '../../../AV_Cleanse_withRJCA/train_all_clean.txt'

lines = open(train_list, 'r').readlines()

missing_files = open('/misc/scratch11/Voxceleb2_Data/missing.txt', 'w')
available_files = open('/misc/scratch11/Voxceleb2_Data/refined_train_clean.txt', 'w')

for line in lines:
    if os.path.isdir(os.path.join(train_path, 'Insightface_Images', line.split()[1][:-4])):
        available_files.write(line)
    else:
        missing_files.write(line)
