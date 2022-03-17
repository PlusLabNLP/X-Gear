#!/bin/bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

MODEL=$1
OUTPUT_DIR=$2

CONFIG_EN="./config/config_ace05_mT5copy-base_en.json"
CONFIG_AR="./config/config_ace05_mT5copy-base_ar.json"
CONFIG_ZH="./config/config_ace05_mT5copy-base_zh.json"

echo "======================"
echo "Predicting for English"
echo "======================"
python ./xgear/evaluate.py --constrained_decode -c $CONFIG_EN -m $MODEL -o $OUTPUT_DIR/en

echo "======================"
echo "Predicting for Arabic"
echo "======================"
python ./xgear/evaluate.py --constrained_decode -c $CONFIG_AR -m $MODEL -o $OUTPUT_DIR/ar

echo "======================"
echo "Predicting for Chinese"
echo "======================"
python ./xgear/evaluate.py --constrained_decode -c $CONFIG_ZH -m $MODEL -o $OUTPUT_DIR/zh
