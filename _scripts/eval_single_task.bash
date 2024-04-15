CUDA_VISIBLE_DEVICES=$1 python simple_main.py \
    --do_sample_eval \
    --save_sample_eval_path $DIR_EVAL/stats.pickle \
    --config_file  $DIR_EVAL/cfg.yaml \
    DATA.SHUFFLE_AFTER_LIMIT True \
    DATA.PROMPT_ARGS.LOAD_RATIONALES False \
    DATA.PROMPT_ARGS.PREDICT_RATIONALE False \
    DATA.RANDOM_DROP_COT_PROMPTS False \
    DATA.SHUFFLE_COT_PROMPTS False \
    DATA.PROMPT_ARGS.USE_RANDOM_TRAIN_PROMPT False \
    POLICY_MODEL.WEIGHT $DIR_EVAL/model_${MODEL_EPOCH}.pth \
    DATA.EVAL_BATCH_SIZE 1 \
    POLICY_MODEL.MAX_INPUT_LENGTH 700 \
    ${@:2}