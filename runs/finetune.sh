#!/bin/sh

echo "fine-tune via train_RLMIL.py"
for i in  0 1 2 3 4; do
  for STAGE in 1 2; do
    python ../train_HeMiCoRe.py \
      --dataset CRC \
      --data_csv path/to/CRC_10.csv \
      --data_split_json path/to/data_split_${i}.json \
      --train_data train \
      --feat_size 1024 \
      --feat_size_ratio 0.5 \
      --preload \
      --train_method finetune \
      --train_stage ${STAGE} \
      --checkpoint_pretrained "path/to/pretrain/stage3/model_best.pth.tar" \
      --T 6 \
      --scheduler CosineAnnealingLR \
      --Cluster 10 \
      --coord_clusters 4 \
      --batch_size 24 \
      --epochs 50 \
      --backbone_lr 0.00005 \
      --fc_lr 0.00001 \
      --patience 10 \
      --arch HGNN \
      --device 0 \
      --base_save_dir ../results/CRC \
      --save_model \
      --exist_ok
  done
done