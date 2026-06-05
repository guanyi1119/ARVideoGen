export MODELS_DIR="/cache/wan_models"
export DEVICE_TYPE="npu"
export PYTHONPATH=$PYTHONPATH:"/use/local/Ascend/ascend-toolkit/latest/python/site-packages":"./"

export HCCL_INTER_HCCS_DISABLE=TRUE
export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_DETERMINISTIC=TRUE
export HCCL_DATA_PARALLEL_OPTIMIZE=TRUE
export TASK_QUEUE_ENABLE=1

python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/models/wan_models/Wan2.1-T2V-1.3B', '/cache/wan_models/Wan2.1-T2V-1.3B')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/models/wan_models/Wan2.1-T2V-14B', '/cache/wan_models/Wan2.1-T2V-14B')"
python -c "import moxing as mox; mox.file.copy('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/models/Self-Forcing/vidprom_filtered_extended.txt', '/cache/vidprom_filtered_extended.txt')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/models/VideoReward', '/cache/VideoReward')"

# ode init model should be put here: /cache/ode_model.pt, like:
# python -c "import moxing as mox; mox.file.copy('obs://yw-ads-model-training-gy1/model-dev/pixelgeek/video-gen-ar/2026/05/08/d64d9e701c134c5a951425f605fc179e/output/train_outputs/causal_forcing_ode_chunkwise_continue/checkpoint_model_004000/model.pt', '/cache/ode_model.pt')"

MASTER_ADDR=$(echo $VC_WORKER_HOSTS | cut -d',' -f1)

torchrun --nproc_per_node=8 --rdzv_conf="timeout=7200" --nnodes=$VC_WORKER_NUM --node_rank=$VC_TASK_INDEX --master_addr $MASTER_ADDR --master_port 12345 train_causal_forcing.py \
    --config_path configs/reward_forcing/reward_forcing_dmd.yaml \
    --logdir train_outputs/reward_forcing_dmd
