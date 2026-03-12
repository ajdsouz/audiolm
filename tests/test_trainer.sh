# !/bin/bash

uv run tests/trainer_test.py --dataset_path='data/' \
    --checkpoint_dir='ckpt/' \
    --logfile_path='logs/test.log' \
    --wandb_project_name='Custom Qwen Training Test - Fabric' \
    --wandb_entity=tororo \
    --wandb_run_name='first run kinda excited' \
    --lr=0.0005 \
    --device='cpu' \
    --num_epochs=1 \
    --batch_size=2 \
    --eval_every=200 \
    --save_every=200 \
    --grad_accumulation_steps=4 \