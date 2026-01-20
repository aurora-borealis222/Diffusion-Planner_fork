#!/bin/bash

###################################
# User Configuration Section
###################################
DATA_ARCHIVE_PATH="$HOME/stadium_main_300.zip"
TRAIN_SET_PATH="$HOME/swarm_data_fixed5"
###################################

python swarm_data_process.py \
  --data_path $DATA_ARCHIVE_PATH \
  --save_path $TRAIN_SET_PATH
