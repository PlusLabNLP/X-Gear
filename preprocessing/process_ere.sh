#!/bin/bash
export PYTHONWARNINGS=ignore

export ERE_PATH="./Dataset/ERE_ES"
export OUTPUT_PATH="../processed_data"
mkdir $OUTPUT_PATH

export TOKENIZER_NAME='mT5'
export PRETRAINED_TOKENIZER_NAME='google/mt5-large'
mkdir $OUTPUT_PATH/ere_es_$TOKENIZER_NAME
python src/process_ere.py -i $ERE_PATH -o $OUTPUT_PATH/ere_es_$TOKENIZER_NAME -s src/splits/ERE-ES -b $PRETRAINED_TOKENIZER_NAME -w 1 -l spanish

#==============================================================#
export ERE_PATH="./Dataset/ERE_EN"
export OUTPUT_PATH="../processed_data"
mkdir $OUTPUT_PATH

export TOKENIZER_NAME='mT5'
export PRETRAINED_TOKENIZER_NAME='google/mt5-large'
mkdir $OUTPUT_PATH/ere_en_$TOKENIZER_NAME
python src/process_ere.py -i $ERE_PATH -o $OUTPUT_PATH/ere_en_$TOKENIZER_NAME -s src/splits/ERE-EN -b $PRETRAINED_TOKENIZER_NAME -w 1 -l english
