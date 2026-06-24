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
# python -c "import moxing as mox; mox.file.copy('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/DMDPrompts/vidprom_filtered_extended.txt', '/cache/vidprom_filtered_extended.txt')"
# python -c "import moxing as mox; mox.file.copy('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/DMDPrompts/vidprom_filtered_extended_switch.txt', '/cache/vidprom_filtered_extended_switch.txt')"
python -c "import moxing as mox; mox.file.copy('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/DMDPrompts/pretrain_30w_15s_20230211_0-5s.txt', '/cache/vidprom_filtered_extended.txt')"
python -c "import moxing as mox; mox.file.copy('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/DMDPrompts/pretrain_30w_15s_20230211_5-10s.txt', '/cache/vidprom_filtered_extended_switch.txt')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/models/VideoReward', '/cache/VideoReward')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/models/Qwen2-VL-2B-Instruct', '/cache/Qwen2-VL-2B-Instruct')"

# pretrained dmd model should be put here: /cache/reward_forcing_dmd.pt

MASTER_ADDR=$(echo $VC_WORKER_HOSTS | cut -d',' -f1)

torchrun --nproc_per_node=8 --rdzv_conf="timeout=7200" --nnodes=$VC_WORKER_NUM --node_rank=$VC_TASK_INDEX --master_addr $MASTER_ADDR --master_port 12345 train_reward_forcing.py \
    --config_path configs/reward_forcing/reward_forcing_dmd_streaming.yaml \
    --logdir train_outputs/reward_forcing_dmd_streaming
