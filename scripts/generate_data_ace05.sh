#!/bin/bash

CONFIG_EN="./config/config_ace05_mT5copy-base_en.json"
CONFIG_AR="./config/config_ace05_mT5copy-base_ar.json"
CONFIG_ZH="./config/config_ace05_mT5copy-base_zh.json"

python ./xgear/generate_data.py -c $CONFIG_EN
python ./xgear/generate_data.py -c $CONFIG_AR
python ./xgear/generate_data.py -c $CONFIG_ZH