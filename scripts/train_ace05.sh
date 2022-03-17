#!/bin/bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

CONFIG="./config/config_ace05_mT5copy-base_en.json"
# CONFIG="./config/config_ace05_mT5copy-base_ar.json"
# CONFIG="./config/config_ace05_mT5copy-base_zh.json"

python ./xgear/train.py -c $CONFIG

