export CUDA_VISIBLE_DEVICES=0

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="$HOME/miniconda3/envs/diffusion_planner_swarm/bin/python" # python path (e.g., "/home/xxx/anaconda3/envs/diffusion_planner/bin/python")

# Set training data path
TRAIN_SET_PATH="$HOME/swarm_data_fixed5" # preprocess data using data_process.sh
TRAIN_SET_LIST_PATH="$HOME/Diffusion-Planner_fork/diffusion_planner_training.json"
###################################

sudo -E $RUN_PYTHON_PATH train_predictor.py \
--train_set  $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH \

