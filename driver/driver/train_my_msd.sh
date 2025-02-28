#!/bin/bash

model=mynet
date=$(date "+%Y%m%d")
num=e1
dataset=MSD
config_file='./config/MSD/configuration.txt'

label_size=10
epochs=9000
#gpu=8
#num=e1_stu_tea
##ws_pre_hds_cp_lb_diff
#exp=ab_w_s_pre_mix_hard_lb_diff
#python3 train_my_w_s_pre_hds_bcp_lb_diff.py --gpu ${gpu} --run-num ${model}_${exp}_${label_size}_${date}_${num} --default-size ${label_size} \
#  --ema-decay 0.99 --bs 2 --model ${model} --epochs ${epochs} --config-file ${config_file} >../../log/${dataset}_${model}_${exp}_${label_size}_${date}_${num}.log 2>&1 &

#shutdown
