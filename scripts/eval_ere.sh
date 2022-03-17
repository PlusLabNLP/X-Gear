#!/bin/bash

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

MODEL=$1
OUTPUT_DIR=$2

CONFIG_EN="./config/config_ere_mT5copy-base_en.json"
CONFIG_ES="./config/config_ere_mT5copy-base_es.json"

echo "======================"
echo "Predicting for English"
echo "======================"
python ./xgear/evaluate.py --constrained_decode -c $CONFIG_EN -m $MODEL -o $OUTPUT_DIR/en

echo "======================"
echo "Predicting for Spanish"
echo "======================"
python ./xgear/evaluate.py --constrained_decode -c $CONFIG_ES -m $MODEL -o $OUTPUT_DIR/es
