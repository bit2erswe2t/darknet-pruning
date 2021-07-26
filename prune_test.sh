#!/bin/bash

# --- start
# make clean && make PRUNE=1 SAT_FEATURE=1 -j4
# bash prune_test.sh

# --- kill
# ps -ef |grep darknet |grep -v "grep"| cut -d' ' -f5 |xargs kill


nohup ./run.sh valid alexnet >>./log/alexnet.log 2>&1 &

nohup ./run.sh mvalid mobilenetv2 >>./log/mobilenetv2.log 2>&1 &

nohup ./run.sh valid densenet201 >>./log/densenet201.log 2>&1 &

nohup ./run.sh valid resnet50 >>./log/resnet50.log 2>&1 &

# layer test
# ./run.sh mpredict mobilenetv2 >./log/mobilenetv2_by_layer.log
# cat ./log/mobilenetv2_by_layer.log |grep "layer feature" |cut -d':' -f2- |cut -d' ' -f2- |xargs -I {} echo {} >>log/analyse_mv2_layer.csv