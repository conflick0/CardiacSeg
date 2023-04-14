# CardiacSeg
## Requirements
* miniconda
* python 3.9
## Install
```shell
./setup.sh
```
## Dataset
* unzip dataset ,and put dataset into `CardiacSeg`.
```
CardiacSeg/
├── README.md
...
└── dataset
```
## Training/Test
### Open Notebook
* open training notebook from `CardiacSeg/exps/main_train.ipynb`.
### Setup Config
#### workspace
* setup absolute path of workspace.
```
workspace = '<workspace>/CardiacSeg'
```
#### model name
* setup model name.
* The model name used in this study is `unetcnx_x3_2_2_a5`. 
* If you want to replace it with other research methods, you can change it to a different model name, such as `swinunetr`, `unetr`, `cotr`, `attention_unet` and `unet`.
```
model_name = 'unetcnx_x3_2_2_a5'
```
### Run
* run all cells, and the final results of the program will display validation scores and inference scores.

![](https://i.imgur.com/ZdmPaNC.png)