#!/bin/bash

# --- start
# make clean && make PRUNE=1 SAT_FEATURE=1 -j4
# bash prune_test.sh

# --- kill
# ps -ef|grep run.sh |grep -v "grep" |cut -d' ' -f4 |xargs kill


nohup ./run.sh valid alexnet >./prune_res/alexnet_0717.log 2>&1 &

nohup ./run.sh mvalid mobilenetv2 >./prune_res/mobilenetv2_0717.log 2>&1 &

nohup ./run.sh valid densenet201 >./prune_res/densenet201_0717.log 2>&1 &

nohup ./run.sh valid resnet50 >./prune_res/resnet50_0717.log 2>&1 &

# layer test
# ./run.sh mpredict mobilenetv2 >./prune_res/mobilenetv2_by_layer.log
# cat ./prune_res/mobilenetv2_by_layer.log |grep "layer feature" |cut -d':' -f2- |cut -d' ' -f2- |xargs -I {} echo {} >>log/analyse_mv2_layer.csv