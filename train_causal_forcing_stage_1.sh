export MODELS_DIR="/data/z00546255/models/wan_models"
export OUTPUT_URL="./"
export DEVICE_TYPE="npu"
export PYTHONPATH=$PYTHONPATH:"/use/local/Ascend/ascend-toolkit/latest/python/site-packages":"./"

export HCCL_INTER_HCCS_DISABLE=TRUE
export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_DETERMINISTIC=TRUE
export HCCL_DATA_PARALLEL_OPTIMIZE=TRUE
export TASK_QUEUE_ENABLE=2

python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/data/CausalForcingData/clean_data', /cache/clean_data')"
python -c "import moxing as mox; mox.file.copy_parallel('obs://yw-pixelgeek-training-data-gy1/01.USERS/z00546255/models/wan_models/Wan2.1-T2V-1.3B', '/cache/wan_models/Wan2.1-T2V-1.3B')"

MASTER_ADDR=$(echo $VC_WORKER_HOSTS | cut -d',' -f1)

torchrun --nproc_per_node=8 train_causal_forcing.py \
    --config_path configs/causal_forcing/ar_diffusion_tf_chunkwise.yaml \
    --logdir train_outputs/causal_forcing_ar_tf_chunkwise