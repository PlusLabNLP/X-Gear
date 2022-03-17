#!/bin/bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

CONFIG="./config/config_ere_mT5copy-base_en.json"
# CONFIG="./config/config_ere_mT5copy-base_es.json"

python ./xgear/train.py -c $CONFIG

