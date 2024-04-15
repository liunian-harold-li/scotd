CUDA_VISIBLE_DEVICES=$1 accelerate launch --main_process_port $2 simple_main.py \
    --config_file $3 \
    TRAIN.CUSTOM_NAME $4 \
    POLICY_MODEL.ACCELERATE_FP16 True \
    POLICY_MODEL.ACCELERATE_MIX_PREDICTION fp16 \
    DATA.CONFIG $5 \
    TRAIN.DATAPOOL_PATH $6 \
    TRAIN.SAVE_INTERVAL 2000 \
    DATA.SHUFFLE_AFTER_LIMIT True \
    ${@:7}