export CUDA_VISIBLE_DEVICES=0

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/home/jovyan/.mlspace/envs/env_marl_TatyanaG_1/bin/python" # python path (e.g., "/home/xxx/anaconda3/envs/diffusion_planner/bin/python")

# Set training data path
TRAIN_SET_PATH="/home/jovyan/AndreiZ/MARL/TatyanaG/swarm_data_fixed5" # preprocess data using data_process.sh
TRAIN_SET_LIST_PATH="/home/jovyan/AndreiZ/MARL/TatyanaG/Diffusion-Planner_fork/diffusion_planner_training.json"
###################################

$RUN_PYTHON_PATH train_predictor.py \
--train_set  $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH \

