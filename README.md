In this work, we present LAVViT: Latent Audio-Visual Vision Transformers for Speaker Verification 

## References
If you find this work useful in your research, please consider citing our work :pencil: and giving a star :star2: :
```bibtex
@INPROCEEDINGS{10888977,
  author={Praveen, R. Gnana and Alam, Jahangir},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={LAVViT: Latent Audio-Visual Vision Transformers for Speaker Verification}, 
  year={2025},
}
```

There are three major blocks in this repository to reproduce the results of our paper. This code uses Mixed Precision Training (torch.cuda.amp). The dependencies and packages required to reproduce the environment of this repository can be found in the `environment.yml` file. 

### Creating the environment
Create an environment using the `environment.yml` file

`conda env create -f environment.yml`

### Text Files

The text files can be found [here](https://drive.google.com/drive/u/0/folders/1NJicFlj9CeNzxvtrOHRIHy6HnoTszro7)
```
train_list :  Train list
val_trials :  Validation trials list
val_list : Validation list
test_trials : VoX1-O trials list
test_list : Vox 1-O list

```


# Table of contents <a name="Table_of_Content"></a>

+ [Preprocessing](#DP) 
    + [Step One: Download the dataset](#PD)
    + [Step Two: Preprocess the visual modality](#PV) 
+ [Training](#Training) 
    + [Training the fusion model](#TE) 
+ [Inference](#R)
    + [Generating the results](#GR)
 
## Preprocessing <a name="DP"></a>
[Return to Table of Content](#Table_of_Content)

### Step One: Download the dataset <a name="PD"></a>
[Return to Table of Content](#Table_of_Content)
Please download the following.
  + The images of Voxceleb1 dataset can be downloaded [here](https://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/) 

### Step Two: Preprocess the visual modality <a name="PV"></a>
[Return to Table of Content](#Table_of_Content)
  + The downloaded images are not properly aligned. So the images are aligned using [Insightface](https://github.com/TadasBaltrusaitis/OpenFace/releases) The preprocessing scripts are provided in preprocessing folder 
  + Please note that it is important to compute mean and standard deviation for audio data (spectrograms) using the command 'sbatch run_mean.sh'

## Training <a name="TE"></a>
[Return to Table of Content](#Table_of_Content)
  + sbatch run_train.sh 

## Inference <a name="GR"></a>
[Return to Table of Content](#Table_of_Content)
  + sbatch run_eval.sh



### 👍 Acknowledgments
Our code is based on [AVCleanse](https://github.com/TaoRuijie/AVCleanse) and [LAVISH] (https://github.com/GenjiB/LAVISH)
