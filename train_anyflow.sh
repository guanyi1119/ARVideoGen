export MODELS_DIR="/data/z00546255/models/wan_models"
export OUTPUT_URL="./"
export DEVICE_TYPE="npu"
export PYTHONPATH=$PYTHONPATH:"/use/local/Ascend/ascend-toolkit/latest/python/site-packages":"./"
export USE_MANUAL_ATTN="1"

export HCCL_INTER_HCCS_DISABLE=TRUE
export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE="AIV"

export HCCL_DETERMINISTIC=TRUE
export HCCL_DATA_PARALLEL_OPTIMIZE=TRUE
export TASK_QUEUE_ENABLE=2

torchrun --nproc_per_node=8 train_anyflow.py \
    --config_path configs/anyflow/train/farwan_causal/pretrain/train_farwan1b_student_shift5_81f_480p_lr5e-5_6k_b32.yml
