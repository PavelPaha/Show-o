# #!/bin/bash

# # Learning rate sweep for MOE training
# # Runs 7 experiments in parallel on GPUs 0-6

# # Улучшенное управление памятью GPU
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # Create logs directory
# mkdir -p logs

# echo "Starting training with lr=1e-6 on GPU 0"
# CUDA_VISIBLE_DEVICES=0 uv run training/train.py \
#     config=configs/showo_mmu_moe.yaml \
#     experiment.name="moe lr 1e-6 bs 4" \
#     moe.enabled=true \
#     optimizer.params.moe_learning_rate=0.000001 \
#     > logs/train_lr_1e-6.log 2>&1 &
# echo "  PID: $!"

# echo "Starting training with lr=2e-6 on GPU 1"
# CUDA_VISIBLE_DEVICES=1 uv run training/train.py \
#     config=configs/showo_mmu_moe.yaml \
#     experiment.name="moe lr 2e-6 bs 4" \
#     moe.enabled=true \
#     optimizer.params.moe_learning_rate=0.000002 \
#     > logs/train_lr_2e-6.log 2>&1 &
# echo "  PID: $!"

# echo "Starting training with lr=5e-6 on GPU 2"
# CUDA_VISIBLE_DEVICES=2 uv run training/train.py \
#     config=configs/showo_mmu_moe.yaml \
#     experiment.name="moe lr 5e-6 bs 4" \
#     moe.enabled=true \
#     optimizer.params.moe_learning_rate=0.000005 \
#     > logs/train_lr_5e-6.log 2>&1 &
# echo "  PID: $!"

# echo "Starting training with lr=1e-5 on GPU 3"
# CUDA_VISIBLE_DEVICES=3 uv run training/train.py \
#     config=configs/showo_mmu_moe.yaml \
#     experiment.name="moe lr 1e-5 bs 4" \
#     moe.enabled=true \
#     optimizer.params.moe_learning_rate=0.00001 \
#     > logs/train_lr_1e-5.log 2>&1 &
# echo "  PID: $!"

# echo "Starting training with lr=2e-5 on GPU 4"
# CUDA_VISIBLE_DEVICES=4 uv run training/train.py \
#     config=configs/showo_mmu_moe.yaml \
#     experiment.name="moe lr 2e-5 bs 4" \
#     moe.enabled=true \
#     optimizer.params.moe_learning_rate=0.00002 \
#     > logs/train_lr_2e-5.log 2>&1 &
# echo "  PID: $!"

# echo "Starting training with lr=5e-5 on GPU 5"
# CUDA_VISIBLE_DEVICES=5 uv run training/train.py \
#     config=configs/showo_mmu_moe.yaml \
#     experiment.name="moe lr 5e-5 bs 4" \
#     moe.enabled=true \
#     optimizer.params.moe_learning_rate=0.00005 \
#     > logs/train_lr_5e-5.log 2>&1 &
# echo "  PID: $!"

# echo "Starting training with lr=1e-4 on GPU 6"
# CUDA_VISIBLE_DEVICES=6 uv run training/train.py \
#     config=configs/showo_mmu_moe.yaml \
#     experiment.name="moe lr 1e-4 bs 4" \
#     moe.enabled=true \
#     optimizer.params.moe_learning_rate=0.0001 \
#     > logs/train_lr_1e-4.log 2>&1 &
# echo "  PID: $!"

# echo ""
# echo "All 7 training processes started!"
# echo "Monitor with: tail -f logs/train_lr_*.log"
# echo "Check running: ps aux | grep train.py"

# wait
# echo "All training runs completed!"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 uv run training/train.py \
    config=configs/showo_mmu_moe.yaml \
    experiment.name="moe lr 1e-5 bs 2 gumbel temp end 0.1" \
    moe.enabled=true \
    moe.use_gumbel=true \
    optimizer.params.moe_learning_rate=0.00001 \
    moe.temp_end=0.1 \
    > logs/train_gumbel_1e-5.log 2>&1 &
echo "  PID: $!"

CUDA_VISIBLE_DEVICES=1 uv run training/train.py \
    config=configs/showo_mmu_moe.yaml \
    experiment.name="moe lr 5e-6 bs 2 gumbel temp end 0.1" \
    moe.enabled=true \
    moe.use_gumbel=true \
    optimizer.params.moe_learning_rate=0.000005 \
    moe.temp_end=0.1 \
    > logs/train_gumbel_5e-6.log 2>&1 &
echo "  PID: $!"