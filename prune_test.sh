#!/bin/bash

# --- start
# bash prune_test.sh
# --- kill
# ps -ef|grep run.sh |grep -v "grep" |cut -d' ' -f4 |xargs kill

nohup ./run.sh valid alexnet >./prune_res/alexnet_0717.log 2>&1 &

nohup ./run.sh mvalid mobilenetv2 >./prune_res/mobilenetv2_0717.log 2>&1 &

nohup ./run.sh valid densenet201 >./prune_res/densenet201_0717.log 2>&1 &

nohup ./run.sh valid resnet50 >./prune_res/resnet50_0717.log 2>&1 &
