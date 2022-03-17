#!/bin/bash

CONFIG_EN="./config/config_ere_mT5copy-base_en.json"
CONFIG_ES="./config/config_ere_mT5copy-base_es.json"

python ./xgear/generate_data.py -c $CONFIG_EN
python ./xgear/generate_data.py -c $CONFIG_ES
