# Hypergraph-based Multiple-instance Learning for Pan-cancer Survival Prediction on Whole Slide Pathology Images
This repo is the PyTorch implementation for the MuRCL described in the paper "MuRCL: Multi-instance Reinforcement Contrastive Learning for Whole Slide Image Classification". 

![fig2](figs/fig2.png)

## Folder structures

```
│  requirements.yaml
│  train_MuRCL.py  # pre-training MuRCL
│  train_RLMIL.py  # training, fine-tuning and linear evaluating RLMIL 
│          
├─models  # the model's architecture
|      __init__.py
│      cl.py
│      clam.py
│      rlmil.py
│      
├─runs  # the training scripts 
│      finetune.sh
│      linear.sh
│      pretrain.sh
│      scratch.sh
│      
├─utils
│      __init__.py
│      datasets.py  # WSI class and function for WSIs
│      general.py   # help function
│      losses.py    # loss function
        
```

## Requirements

environment.yaml


## Datasets

### Download

> TCGA: Use [GDC data portal](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Getting_Started/) with a manifest file and configuration file.

### WSI Processing

1. `cd wsi_processing`

2. Tile all the WSI into patches

   ```shell
   python create_patches.py --slide_dir /dir/to/silde --save_dir /save/dir/patch --overview --save_mask --wsi_format .tif --overview_level 3
   e.g.  python create_patches.py --slide_dir /media/oasis/DATA_1/survival_prediction/KIRC/slide --save_dir /media/oasis/DATA_1/survival_prediction/KIRC --overview --save_mask --wsi_format .svs --overview_level 2
   ```shell
    python extract_features_multiprocess.py --patch_dir /media/oasis/DATA/survival_prediction/patch --image_encoder resnet18 --device 0 --coord coord
   ```

4. Clustering patch features

   ```shell
   python features_clustering.py --feat_dir /dir/to/patch/features --num_clusters 10
   ```
5. visulize clustering distribution

   ```shell
      python visual_clustering.py --overview_level 2
    ```

### Data Organization

The format of  input csv file:

|  case_id   |           features_filepath            | label |            clusters_filepath            |          clusters_json_filepath          |
| :--------: | :------------------------------------: | :---: | :-------------------------------------: | :--------------------------------------: |
| normal_001 | /path/to/patch_features/normal_001.npz |   0   | /path/to/cluster_indices/normal_001.npz | /path/to/cluster_indices/normal_001.json |
|    ...     |                  ...                   |  ...  |                   ...                   |                   ...                    |

> **case_id**: [str] the index for each WSI. 
>
> **features_filepath**: [str] the .npz file path for each WSI, this .npz file contains several keywords as follows: 
>
> - filename: [str] case_id. 
> - img_features: [numpy.ndarray] the all patch's features as a numpy.ndarray, the shape is (num_patches, dim_features), like (1937, 512). 
>
> **label**: [int] the label of the WSI. 
>
> **clusters_filepath**: [str] this .npz file indicates the clustering category corresponding to each patch in WSI. It contaions several keywords as follows:
>
> - filename: [str] case_id.
> - features_cluster_indices: [numpy.ndarray] This array represents the clustering category of each patch feature in WSI, it's shape is (num_pathces, 1). 
>
> **clusters_json_filepath**: [str] This JSON file represents the patch index contained in each category of clustering, like:
>
> ```json
> [
>     [0, 30, 57, 58, 89, 113, 124, 131, ...],
>     [11, 13, 22, 25, 26, 34, 35, 45, 49, 50, 51, ...],
>     ...
>     [1, 8, 15, 16, ...]
> ]
> ```
>
> Each list represents a category.

## Pre-training

pre-training by hypergraph-level contrastive learning. 

```shell
cd runs
sh pretrain.sh
```

## Training from scratch, fine-tuning, and exporting the desicion-making patches

evaluation of our proposed framework MuRCL. 

```shell
cd runs
# training from scatch with the initail weights, does not load pre-trained weights
sh scratch.sh
# fine-tuning with pre-training weights 
sh fintune.sh
# exporting the desicion-making patches by testing with the weights of the target stage
sh get_key_patch.sh
```

## Visualization

This code could create the distribution map mentioned in our paper.

```shell
cd my_utils
python create_heatmaps.py
```

<img src="/figs/tumor_006_pretrain.png" alt="tumor_006_pretrain" width="400" align="middle" /><img src="/figs/tumor_006_finetune.png" alt="tumor_006_finetune" width="400" align="middle" /> 



## Training on your own datasets

1. You can simply process your own dataset into a format acceptable to our code, see [WSI Processing](###WSI Processing) and [Data Organization](###Data Organization). 
2. Then modify the input parameters of the training script in the runs directory. 

