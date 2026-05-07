export MODELS_DIR="/data/z00546255/models/wan_models"
export OUTPUT_URL="./"
export DEVICE_TYPE="npu"
export USE_NPU_FLEX_ATTENTION_VERSION="1"
export PYTHONPATH=$PYTHONPATH:"/use/local/Ascend/ascend-toolkit/latest/python/site-packages":"./"

export HCCL_INTER_HCCS_DISABLE=TRUE
export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_DETERMINISTIC=TRUE
export HCCL_DATA_PARALLEL_OPTIMIZE=TRUE
export TASK_QUEUE_ENABLE=2

torchrun --nproc_per_node=8 train_causvid.py \
    --config_path configs/causvid/causal_ode_init.yaml