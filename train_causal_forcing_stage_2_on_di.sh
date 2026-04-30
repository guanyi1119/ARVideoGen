export MODELS_DIR="/cache/wan_models"
export DEVICE_TYPE="npu"
export PYTHONPATH=$PYTHONPATH:"/use/local/Ascend/ascend-toolkit/latest/python/site-packages":"./"

export HCCL_INTER_HCCS_DISABLE=TRUE
export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_DETERMINISTIC=TRUE
export HCCL_DATA_PARALLEL_OPTIMIZE=TRUE
export TASK_QUEUE_ENABLE=1

mkdir /cache/CausalForcingData
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/ODE6KCausal_chunkwise_0', '/cache/CausalForcingData/ODE6KCausal_chunkwise_0')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/ODE6KCausal_chunkwise_1', '/cache/CausalForcingData/ODE6KCausal_chunkwise_1')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/ODE6KCausal_chunkwise_2', '/cache/CausalForcingData/ODE6KCausal_chunkwise_2')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/ODE6KCausal_chunkwise_3', '/cache/CausalForcingData/ODE6KCausal_chunkwise_3')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/ODE6KCausal_chunkwise_4', '/cache/CausalForcingData/ODE6KCausal_chunkwise_4')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/ODE6KCausal_chunkwise_5', '/cache/CausalForcingData/ODE6KCausal_chunkwise_5')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/ODE6KCausal_chunkwise_6', '/cache/CausalForcingData/ODE6KCausal_chunkwise_6')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/ODE6KCausal_chunkwise_7', '/cache/CausalForcingData/ODE6KCausal_chunkwise_7')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/ODE6KCausal_chunkwise_8', '/cache/CausalForcingData/ODE6KCausal_chunkwise_8')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/ODE6KCausal_chunkwise_9', '/cache/CausalForcingData/ODE6KCausal_chunkwise_9')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/ODE6KCausal_chunkwise_10', '/cache/CausalForcingData/ODE6KCausal_chunkwise_10')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/ODE6KCausal_chunkwise_11', '/cache/CausalForcingData/ODE6KCausal_chunkwise_11')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/ODE6KCausal_chunkwise_12', '/cache/CausalForcingData/ODE6KCausal_chunkwise_12')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/ODE6KCausal_chunkwise_13', '/cache/CausalForcingData/ODE6KCausal_chunkwise_13')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/ODE6KCausal_chunkwise_14', '/cache/CausalForcingData/ODE6KCausal_chunkwise_14')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/models/wan_models/Wan2.1-T2V-1.3B', '/cache/wan_models/Wan2.1-T2V-1.3B')"

MASTER_ADDR=$(echo $VC_WORKER_HOSTS | cut -d',' -f1)

torchrun --nproc_per_node=8 --rdzv_conf="timeout=7200" --nnodes=$VC_WORKER_NUM --node_rank=$VC_TASK_INDEX --master_addr $MASTER_ADDR --master_port 12345 train_causal_forcing.py \
    --config_path configs/causal_forcing/causal_ode_chunkwise.yaml \
    --logdir train_outputs/causal_forcing_ode_chunkwise
