#!/bin/bash
model=mynet
date=$(date "+%Y%m%d")
num=e1
dataset=Fistula
config_file='./config/Fistula/configuration.txt'
echo ${date}

label_size=36
epochs=333
#gpu=1
#num=e1_hds_change
#exp=pre_hds_cp_lb_diff
#python3 train_my_w_s_pre_hds_bcp_lb_diff.py --gpu ${gpu} --run-num ${model}_${exp}_${label_size}_${date}_${num} --default-size ${label_size} \
#  --ema-decay 0.99 --bs 2 --model ${model} --epochs ${epochs} --config-file ${config_file} >../../log/${dataset}_${model}_${exp}_${label_size}_${date}_${num}.log 2>&1 &
#shutdown
