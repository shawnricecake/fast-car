#!/bin/sh

# wget https://huggingface.co/spaces/OpenGVLab/InternGPT/resolve/main/model_zoo/grit_b_densecap_objectdet.pth -P /home/xuans/.cache/vbench/grit_model
# pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Define the video path
# video_path="/mnt/localssd/vila/generated-videos/vila-u-original/"
# video_path="/mnt/localssd/vila/generated-videos/sparse-0.3/"
video_path="/mnt/localssd/vila/generated-videos/sparse-32x-2/"
# video_path="/mnt/localssd/vila/generated-videos/sparse-whole_mlp-0.7/"
# video_path="/mnt/localssd/vila/generated-videos/sparse-non_uniform-0.8/"
# video_path="/mnt/localssd/vila/generated-videos/sparse_attn-local_16/"

# output_path="evaluation_results-sp0.3"
output_path="evaluation_results-sp-32x-2"
# output_path="evaluation_results-whole_mlp-sp0.7"
# output_path="evaluation_results-non_uniform-sp0.8"
# output_path="evaluation_results-spattn_local_16"

export CUDA_VISIBLE_DEVICES=1

# Loop over the dimensions directly
# for dimension in subject_consistency background_consistency temporal_flickering motion_smoothness dynamic_degree aesthetic_quality imaging_quality object_class multiple_objects human_action color spatial_relationship scene temporal_style appearance_style overall_consistency
for dimension in human_action
do
    echo "Evaluating dimension: $dimension"
    vbench evaluate --dimension "$dimension" --videos_path "$video_path" --output_path $output_path
done
echo "All evaluations completed."


