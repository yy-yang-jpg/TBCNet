<div align="center">
<h2>
TBCNet: Trunk-branch Contrastive Network with Multi-view Deformable Aggregation for Multi-view Action Recognition
</h2>
</div>

## Contents
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Training and Testing](#training-and-testing)

## Requirements
> - Python >= 3.12.2
> - PyTorch >= 1.12.1
> - Platforms: Ubuntu 22.04, CUDA 12.1
> - conda virtual environment is recommended. 
Run `conda create -n TBCNet python=3.12.2`
> - Run `pip install -r requirements.txt`.

## Data Preparation

### Download datasets

#### There are 3 datasets to download:

- NTU-RGB+D 60
- NTU-RGB+D 120
- N-UCLA
- PKU-MMD

#### NTU-RGB+D 60 and NTU-RGB+D 120

1. Request dataset [here](https://rose1.ntu.edu.sg/dataset/actionRecognition)
2. Download the RGB-only datasets:
   1. Extract the files of NTU-RGB+D 60 to `./data/dataset/ntu-rgb/ntu-60/`
   2. Extract the files of NTU-RGB+D 120 to `./data/dataset/ntu-rgb/ntu-120/`

#### N-UCLA

1. Download the RGB-only dataset from [here](https://wangjiangb.github.io/my_data.html)
2. Move video files to `./data/dataset/N-UCLA/`

#### PKU-MMD

1. Download the RGB-only dataset from [here](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html?aimglfkfkfcjmopp)
2. Move video files to `./data/dataset/PKU-MMD/`

### Data Processing

#### Generating hdf5 files

```
cd ./data/data_label
# transform .avi to .hdf5
python data_pre.py
```
The generated files are placed into `./data/dataset_hdf5/`

## Training and Testing

### Edit configuration file
Choose three important parameters: traintype/testtype, dataset, and loss. 
- The traintype/testtype includes CS, CV, and CSet(CS=Cross-Subject, CV=Cross-View, CSet=Cross-Setting). 
- The dataset includes NTU-RGB+D 60, NTU-RGB+D 120, and PKU-MMD. 
- The loss includes c(Cross-Entropy), cv(Cross-Entropy + trunk-branch contrastive Loss), and cvl(Cross-Entropy + weighted trunk-branch contrastive Loss).
The configuration file are placed into `./configs/`

### Train
```bash
python main.py --config ./configs/train/ntu-60_train_CS.yaml
```
### Test
```bash
python test.py --config ./configs/test/ntu-60_test_CS.yaml