#!/bin/bash

###################################
# User Configuration Section
###################################
DATA_ARCHIVE_PATH="$HOME/stadium_main_300.zip"
SAVE_PATH="$HOME/swarm_data_centered"
###################################

python swarm_data_process.py \
  --data_path $DATA_ARCHIVE_PATH \
  --save_path $SAVE_PATH
