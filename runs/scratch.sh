#!/bin/sh

echo "training from scratch via train_RLMIL.py"
for i in 0 1 2 3 4 ; do
  for STAGE in 1 2; do
    python ../train_RLMIL.py \
      --dataset CRC \
      --data_csv /media/oasis/DATA_1/survival_prediction/CRC/train_data/CRC_10.csv \
      --data_split_json /media/oasis/DATA_1/survival_prediction/CRC/train_data/cross_5_fold/data_split_${i}.json \
      --train_data train \
      --feat_size 1024 \
      --preload \
      --train_method scratch \
      --train_stage ${STAGE} \
      --T 6 \
      --scheduler CosineAnnealingLR \
      --Cluster 10 \
      --coord_clusters 4 \
      --batch_size 24 \
      --epochs 50 \
      --backbone_lr 0.0001 \
      --fc_lr 0.00001 \
      --arch HGNN \
      --patience 10 \
      --device 0 \
      --base_save_dir ../results/new_CRC_cross/feat \
      --save_model \
      --exist_ok

  done
#  python ../train_RLMIL.py \
#  --dataset KIRC \
#  --data_csv /media/oasis/DATA_1/survival_prediction/KIRC/train_data/KIRC_10.csv \
#  --data_split_json /media/oasis/DATA_1/survival_prediction/KIRC/train_data/5_fold/data_split_${i}.json \
#  --train_data train \
#  --feat_size 1024 \
#  --preload \
#  --train_method scratch \
#  --train_stage 3 \
#  --T 6 \
#  --scheduler CosineAnnealingLR \
#  --Cluster 10 \
#  --coord_clusters 4 \
#  --batch_size 24 \
#  --epochs 50 \
#  --backbone_lr 0.00001 \
#  --fc_lr 0.00001 \
#  --arch HGNN \
#  --patience 10 \
#  --base_save_dir ../results/KIRC/feat_n_coord \
#  --device 0 \
#  --save_model \
#  --exist_ok
done