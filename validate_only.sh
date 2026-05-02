python validate_only.py \
  --val_set "/home/jovyan/AndreiZ/MARL/TatyanaG/swarm_data_npy" \
  --val_set_list val.json \
  --resume_model_path ./training_log/diffusion-planner-training/2026-05-01-22:04:37/latest.pth \
  #--batch_size 64 \
  --num_workers 8 \
  --quick_val_experiments 5
