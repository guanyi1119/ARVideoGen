export MODELS_DIR="/data/z00546255/models/wan_models"
export DEVICE_TYPE="npu"
export PYTHONPATH=$PYTHONPATH:"/use/local/Ascend/ascend-toolkit/latest/python/site-packages":"./"

export ASCEND_RT_VISIBLE_DEVICES=0

python inference_causvid.py \
    --mode bidirectional \
    --config_path configs/causvid/bidirectional_dmd_init.yaml \
    --checkpoint_path /cache/train_outputs_on_di/causvid_bidirectional_dmd_init/checkpoint_model_008000 \
    --prompt_file_path archive/Self-Forcing/prompts/vbench/all_dimension_extended.txt \
    --output_folder /cache/inference_outputs/inference_causvid_bidirectional_dmd_init_test
