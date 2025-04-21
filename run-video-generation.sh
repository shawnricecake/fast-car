# Image Generation

model_path="/mnt/localssd/vila/checkpoints/vila-u-7b-256"

save_path="/sensei-fs/users/xuans/code/unified-understanding-generation/efficient-unified-autoregressive/zoutputs-6/"

# prompt="Fireworks in the air."
prompt="a baseball glove on the right of a tennis racket, front view"

cache_percent_list_mlp="[-4]*32"

CUDA_VISIBLE_DEVICES=5 \
python3 -u inference.py \
        --model_path $model_path \
        --prompt "$prompt" \
        --save_path $save_path \
        --video_generation True \
        --cache_percent_list_mlp $cache_percent_list_mlp \

