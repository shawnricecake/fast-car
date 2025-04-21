cd ..

model_path="/mnt/localssd/vila/checkpoints/vila-u-7b-256"

# save_path="/sensei-fs/users/xuans/code/unified-understanding-generation/efficient-unified-autoregressive/zoutputs/vila-u-original"
# save_path="/sensei-fs/users/xuans/code/unified-understanding-generation/h100-generation-results/original-baseline/"
# save_path="/mnt/localssd/vila/generated-videos/sparse-0.1"
# save_path="/mnt/localssd/vila/generated-videos/sparse-0.2"
# save_path="/mnt/localssd/vila/generated-videos/sparse-0.3"
# save_path="/mnt/localssd/vila/generated-videos/sparse-0.4"
# save_path="/mnt/localssd/vila/generated-videos/sparse-0.5"
# save_path="/mnt/localssd/vila/generated-videos/sparse-0.6"
# save_path="/mnt/localssd/vila/generated-videos/sparse-0.7"
# save_path="/mnt/localssd/vila/generated-videos/sparse-0.8"
# save_path="/mnt/localssd/vila/generated-videos/sparse-0.87"

# save_path="/mnt/localssd/vila/generated-videos/sparse-32x-1"
# save_path="/mnt/localssd/vila/generated-videos/sparse-32x-8"

# save_path="/mnt/localssd/vila/generated-videos/sparse-whole_mlp-0.2"
# save_path="/mnt/localssd/vila/generated-videos/sparse-whole_mlp-0.1"
# save_path="/mnt/localssd/vila/generated-videos/sparse-whole_mlp-0.7"

# save_path="/mnt/localssd/vila/generated-videos/sparse-non_uniform-0.7"

save_path="/mnt/localssd/vila/generated-videos/sparse-sparseattn_local_128-32x-1"

cache_percent_list_mlp="[-1]*32"

split=5

CUDA_VISIBLE_DEVICES=$split \
python3 -u vbench-video_generation.py \
        --model_path $model_path \
        --save_path $save_path \
        --split $split \
        --video_generation True \
        --cache_percent_list_mlp $cache_percent_list_mlp \
