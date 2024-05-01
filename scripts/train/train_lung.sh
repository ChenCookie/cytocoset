#!/bin/bash

h_dim=256
in_dim=31
pool=max
out_pool=max
nblock=1
ncell_list=("1024")
nsubset=20000
co_factor=5


# train_fcs_info=/playpen-ssd/chijane/data_pree/data_pree_4/pree_fcs/train/train_labels.csv
# test_fcs_info=/playpen-ssd/chijane/data_pree/data_pree_4/pree_fcs/test/test_labels.csv
# markerfile=/playpen-ssd/chijane/data_pree/data_pree_4/pree_fcs/marker.csv


file_base=/playpen-ssd/chijane/data_lung/data_lung_1
train_fcs_info=/lung_fcs/all/train_labels_
test_fcs_info=/lung_fcs/all/test_labels_
file_end=.csv
markerfile=/marker.csv



seed_list=("1")
alpha_list=("0.8")
same_list=("0.2")
diff_list=("0.2")
name_end_list=("4" "44" "444" "4444" "44444" "5" "55")
lr=0.0001
wts_decay=0.001
batch_size=200
n_epochs=10
# log_dir=/playpen-ssd/chijane/data_pree/data_pree_4/pree_fcs
log_dir=/playpen-ssd/chijane/data_lung/data_lung_1
log_interval=1
save_interval=1
patience=5

bin_file=/playpen-ssd/chijane/cytoset/train.py
gpu=$0

for name_end in ${name_end_list[@]}
do
  for ncell in ${ncell_list[@]}
  do
    for seed in ${seed_list[@]}
    do
      for alpha in ${alpha_list[@]}
      do
        for same_t in ${same_list[@]}
        do
          for diff_t in ${diff_list[@]}
          do
            echo "ncell: $ncell, seed: $seed, alpha: $alpha"
            CUDA_VISIBLE_DEVICES=0 python ${bin_file} \
              --in_dim ${in_dim} \
              --h_dim ${h_dim} \
              --pool ${pool} \
              --out_pool ${out_pool} \
              --nblock ${nblock} \
              --ncell ${ncell} \
              --nsubset ${nsubset} \
              --co_factor ${co_factor} \
              --train_fcs_info ${file_base}${train_fcs_info}${name_end}${file_end} \
              --test_fcs_info ${file_base}${test_fcs_info}${name_end}${file_end} \
              --markerfile ${file_base}/lung_fcs${markerfile} \
              --lr ${lr} \
              --wts_decay ${wts_decay} \
              --generate_valid \
              --test_rsampling \
              --shuffle \
              --batch_size ${batch_size} \
              --n_epochs ${n_epochs} \
              --log_dir ${log_dir}${name_end}/lung_fcs_${ncell}_${nblock}_${seed} \
              --log_interval ${log_interval} \
              --save_interval ${save_interval} \
              --patience ${patience} \
              --name_end ${name_end} \
              --same_t ${same_t} \
              --diff_t ${diff_t} \
              --alpha ${alpha} \
              --seed ${seed}
          done
        done
      done
    done
  done
done

exit
