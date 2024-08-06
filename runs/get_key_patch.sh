#!/bin/sh

echo "fine-tune via train_RLMIL.py"
for i in 2 ; do
  for STAGE in 3 ; do
    python ../train_RLMIL.py \
      --dataset CRC \
      --data_csv path/to/CRC_10.csv \
      --data_split_json path/to/data_split_${i}.json \
      --train_data train \
      --feat_size 1024 \
      --feat_size_ratio 0.5 \
      --preload \
      --train_method finetune \
      --train_stage ${STAGE} \
      --checkpoint_pretrained "path/to/target_stage/model_best.pth.tar" \
      --T 6 \
      --scheduler CosineAnnealingLR \
      --Cluster 10 \
      --coord_clusters 4 \
      --batch_size 24 \
      --epochs 50 \
      --backbone_lr 0.0001 \
      --fc_lr 0.00005 \
      --patience 10 \
      --arch HGNN \
      --device 0 \
      --base_save_dir ../results/CRC/for_visualization \
      --save_model \
      --exist_ok \
      --just_test True
  done
done