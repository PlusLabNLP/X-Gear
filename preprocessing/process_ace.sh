#!/bin/bash

export ACE_PATH="./Dataset/ace_2005_td_v7/data/"
export OUTPUT_PATH="../processed_data"
mkdir $OUTPUT_PATH

export TOKENIZER_NAME='mT5'
export PRETRAINED_TOKENIZER_NAME='google/mt5-large'
mkdir $OUTPUT_PATH/ace05_zh_$TOKENIZER_NAME
python src/process_ace.py -i $ACE_PATH -o $OUTPUT_PATH/ace05_zh_$TOKENIZER_NAME -s src/splits/ACE05-ZH -b $PRETRAINED_TOKENIZER_NAME -w 1 -l chinese

#===============================================================================#
export ACE_XU_PATH="./Dataset/ace_2005_Xuetal/en/json"
export OUTPUT_PATH="../processed_data"
mkdir $OUTPUT_PATH

export TOKENIZER_NAME='mT5'
export PRETRAINED_TOKENIZER_NAME='google/mt5-large'
mkdir $OUTPUT_PATH/ace05_en_$TOKENIZER_NAME
for SET in train dev test
do
    python src/process_ace_xuetal.py -i $ACE_XU_PATH/${SET}.json -o $OUTPUT_PATH/ace05_en_$TOKENIZER_NAME/${SET}.json -b $PRETRAINED_TOKENIZER_NAME -w 1 -l english
done
#===============================================================================#