#!/bin/sh

echo "pre-training via pretrain_HeMiCoRe.py"
for i in 4 ; do
  for STAGE in 1 2; do
    python ../pretrain_HeMiCoRe.py \
      --dataset CRC \
      --data_csv path/to/CRC_10.csv \
      --data_split_json path/to/data_split_${i}.json \
      --feat_size 1024 \
      --preload \
      --train_stage ${STAGE} \
      --T 6 \
      --scheduler CosineAnnealingLR \
      --Cluster 10 \
      --coord_clusters 4 \
      --batch_size 24 \
      --epochs 50 \
      --backbone_lr 0.0001 \
      --fc_lr 0.00005 \
      --patience 10 \
      --base_save_dir ../results/CRC \
      --arch HGNN \
      --device 0 \
      --exist_ok
  done
  python ../pretrain_HeMiCoRe.py \
    --dataset CRC \
    --data_csv path/to/CRC_10.csv \
    --data_split_json path/to/data_split_${i}.json \
    --feat_size 1024 \
    --preload \
    --train_stage 3 \
    --T 6 \
    --scheduler CosineAnnealingLR \
    --Cluster 10 \
    --coord_clusters 4 \
    --batch_size 24 \
    --epochs 50 \
    --backbone_lr 0.00005 \
    --fc_lr 0.00001 \
    --patience 10 \
    --base_save_dir ../results/CRC \
    --arch HGNN \
    --device 0 \
    --exist_ok
done
