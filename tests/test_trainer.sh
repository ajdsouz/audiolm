# !/bin/bash

uv run tests/trainer_test.py --dataset_path='data/' \
    --checkpoint_dir='ckpt/' \
    --logfile_path='logs/test.log' \
    --wandb_project_name='Custom Qwen Training Test - Fabric' \
    --wandb_entity=tororo \
    --wandb_run_name='bf16 test' \
    --lr=0.0005 \
    --device='cpu' \
    --precision='32-true' \
    --num_epochs=1 \
    --batch_size=2 \
    --eval_every=50 \
    --save_every=250 \
    --grad_accumulation_steps=4 \