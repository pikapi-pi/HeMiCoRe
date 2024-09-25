# Hypergraph-based Multiple-instance Learning for Annotation-free Pan-cancer Survival Prediction on Whole Slide Pathology Images
This repo is the PyTorch implementation for the HeMiCoRe described in the paper "Hypergraph-based Multiple-instance Learning for Annotation-free Pan-cancer Survival Prediction on Whole Slide Pathology Images".

![fig2](figs/12_44.png)

## Folder structures

```
│  environment.yaml
│  pretrain_HeMiCoRe.py  # pre-training HeMiCoRe
│  train_HeMiCoRe.py  # training, fine-tuning and evaluating HeMiCoRe
├─HGNN
|      H_GNN.py
|      hypergraph_util.py
|      layers.py
|            
├─models  # the model's architecture
|      __init__.py
│      cl.py
│      rlmil.py
│      
├─runs  # the training scripts 
│      finetune.sh
|      get_key_patch.sh
│      pretrain.sh
│      scratch.sh
│      
├─utils
│      __init__.py
│      datasets.py  # WSI class and function for WSIs
│      general.py   # help function
│      losses.py    # loss function
        
```
## OS Requirements

The code was designed on Ubuntu 22.04 LTS

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
   ```

3. Extract patch features

   ```shell
    python extract_features_multiprocess.py --patch_dir /dir/to/patch --image_encoder resnet18 --device 0 --coord coord
   ```

4. Clustering patch features

   ```shell
   python features_clustering.py --feat_dir /dir/to/patch/features --num_clusters 10
   ```

### Data Organization

The format of  input csv file:

|  case_id   |                features_filepath                | label |           clusters_filepath                      |               clusters_json_filepath              |
| :--------: |:-----------------------------------------------:| :---: |:------------------------------------------------:|:-------------------------------------------------:|
| normal_001 | /path/to/patch_features/2014_00322-3-HE-DX1.npz |   0   | /path/to/cluster_indices/2014_00322-3-HE-DX1.npz | /path/to/cluster_indices/2014_00322-3-HE-DX1.json |
|    ...     |                       ...                       |  ...  |                       ...                        |                        ...                        |

> **case_id**: [str] the index for each WSI. 
>
> **features_filepath**: [str] the .npz file path for each WSI, this .npz file contains several keywords as follows: 
>
> - filename: [str] case_id. 
> - img_features: [numpy.ndarray] the all patch's features as a numpy.ndarray, the shape is (num_patches, dim_features), like (9004, 512). 
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
>     [55, 68, 86, 112, 126, 131, 145, 149, ...],
>     [0, 1, 5, 12, 27, 53, 74, 92, 134, 178, 185, ...],
>     ...
>     [6, 19, 30, 36, 45, 48, 61, 77, 78, ...]
> ]
> ```
>
> Each list represents a cluster, and a WSI is divide into multiple subregions. Based on the affiliation between patches and subregions, a WSI can be regarded as a cluster-constrained hypergraph. MIL can be regarded as a form of graph          augmentation.

## Pre-training

pre-training by hypergraph-level contrastive learning. Note: maybe you need to change the final row of code in utils/datasets.py.

```shell
cd runs
sh pretrain.sh
```

## Training from scratch, fine-tuning, and exporting the desicion-making patches

evaluation of our proposed framework HeMiCoRe. Note: maybe you need to change the final row of code in utils/datasets.py.

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
python visualization.py
```


## Training on your own datasets

1. You can simply process your own dataset into a format acceptable to our code, see [WSI Processing](###WSI Processing) and [Data Organization](###Data Organization). 
2. Then modify the input parameters of the training script in the runs directory. 

-
