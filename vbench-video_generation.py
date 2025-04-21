import argparse
import cv2
import numpy as np
import os
import random
import json
import time
import ast
from datetime import datetime
# Xuan: Set the random seed for reproducibility
random.seed(42)
import torch
torch.manual_seed(42)

import vila_u


def parse_list(s: str):
    return eval(s)

def save_image(response, path):
    os.makedirs(path, exist_ok=True)
    for i in range(response.shape[0]):
        image = response[i].permute(1, 2, 0)
        image = image.cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path, f"image_{i}.png"), image)


def save_video(response, path):
    os.makedirs(path, exist_ok=True)
    for i in range(response.shape[0]):
        video = response[i].permute(0, 2, 3, 1)
        video = video.cpu().numpy().astype(np.uint8)
        video = np.concatenate(video, axis=1)
        video = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path, f"video_{i}.png"), video)

# Xuan: Save video as mp4
def save_video_mp4(response, path, prompt, pre=0, fps=1):
    # os.makedirs(path, exist_ok=True)
    
    # Iterate over each video in the batch
    for i in range(response.shape[0]):
        # Permute tensor from [num_frames, channels, height, width] to [num_frames, height, width, channels]
        video_tensor = response[i].permute(0, 2, 3, 1)
        
        # Convert to numpy array in uint8 format
        video_np = video_tensor.cpu().numpy().astype(np.uint8)
        
        # Get the shape of individual video frames
        num_frames, height, width, channels = video_np.shape
        
        # Define output video file path and initialize VideoWriter
        video_path = os.path.join(path, f"{prompt}-{pre+i}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Write each frame into the video file
        for frame in video_np:
            # Convert the frame from RGB (common for tensor data) to BGR (used by OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            
        writer.release()

def save_video_mp4_batch1(response, path, fps=1):
    # os.makedirs(path, exist_ok=True)
    
    # Iterate over each video in the batch
    for i in range(response.shape[0]):
        # Permute tensor from [num_frames, channels, height, width] to [num_frames, height, width, channels]
        video_tensor = response[i].permute(0, 2, 3, 1)
        
        # Convert to numpy array in uint8 format
        video_np = video_tensor.cpu().numpy().astype(np.uint8)
        
        # Get the shape of individual video frames
        num_frames, height, width, channels = video_np.shape
        
        # Define output video file path and initialize VideoWriter
        # video_path = os.path.join(path, f"video-{i}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        
        # Write each frame into the video file
        for frame in video_np:
            # Convert the frame from RGB (common for tensor data) to BGR (used by OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            
        writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    ### image/video understanding arguments
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.9, help="The value of temperature for text generation.")
    parser.add_argument("--top_p", type=float, default=0.6, help="The value of top-p for text generation.")
    ### image and video generation arguments
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--video_generation", type=bool, default=False)
    parser.add_argument("--cfg", type=float, default=3.0, help="The value of the classifier free guidance for image generation.")
    parser.add_argument("--save_path", type=str, default="generated_images/")
    parser.add_argument("--generation_nums", type=int, default=1)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--num_split", type=int, default=8)
    parser.add_argument("--dim", type=str, default=None)
    parser.add_argument("--cache_percent_list_mlp", type=parse_list, default=[None]*32)
    # parser.add_argument("--all_prompts_json", type=str, default=None)
    args = parser.parse_args()

    dimensions=["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "object_class", "multiple_objects", "color", "spatial_relationship", "scene", "temporal_style", "overall_consistency", "human_action", "temporal_flickering", "motion_smoothness", "dynamic_degree", "appearance_style"]
    if args.dim is not None and args.dim not in dimensions:
        raise ValueError(f"Invalid dimension: {args.dim}. Choose from {dimensions}.")

    if args.model_path is not None:
        model = vila_u.load(args.model_path)
    else:
        raise ValueError("No model path provided!")

    print("cache_percent_list_mlp: ", args.cache_percent_list_mlp)
    print("start time: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    all_prompts = json.load(open("prompts/VBench_full_info-5.json"))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    for idx, e in enumerate(all_prompts):
        if idx < int(len(all_prompts) * args.split / args.num_split) or \
        idx >= int(len(all_prompts) * (args.split + 1) / args.num_split):
            continue

        print(f"Processing {idx+1}/{len(all_prompts)}")

        current_dim = e['dimension']
        if args.dim is not None and args.dim not in current_dim:
            continue

        sample_times = 5

        print(f"Sample {sample_times}/{sample_times}")

        current_prompt = e['prompt_en']
        current_video_name = current_prompt + "-0" + ".mp4"
        current_save_path = os.path.join(args.save_path, current_video_name)

        if os.path.exists(current_save_path):
            print(f"File already exists: {current_save_path}")
            continue
        print(f"Generating video for prompt: {current_prompt}")

        current_prompt = [current_prompt] * 5
        response = model.generate_video_content_batch(
            current_prompt, args.cfg, args.generation_nums,
            cache_percent_list_mlp=args.cache_percent_list_mlp
        )
        save_video_mp4(response, args.save_path, current_prompt[0])

        for i_layer in range(len(model.llm.model.layers)):
            model.llm.model.layers[i_layer].self_attn.query_cache = []
            model.llm.model.layers[i_layer].self_attn.cache_if_skip_k_v_projection = []
            model.llm.model.layers[i_layer].cache_attn_input = []
            model.llm.model.layers[i_layer].cache_attn = []
            model.llm.model.layers[i_layer].cache_mlp_input = []
            model.llm.model.layers[i_layer].cache_mlp = []
            model.llm.model.layers[i_layer].cache_if_skip_mlp = []
    

    all_prompts = json.load(open("prompts/VBench_full_info-25.json"))

    for idx, e in enumerate(all_prompts):
        if idx < int(len(all_prompts) * args.split / args.num_split) or \
        idx >= int(len(all_prompts) * (args.split + 1) / args.num_split):
            continue

        print(f"Processing {idx+1}/{len(all_prompts)}")

        current_dim = e['dimension']
        if args.dim is not None and args.dim not in current_dim:
            continue

        sample_times = 25

        for sample_idx in range(5):
            print(f"Sample {(sample_idx+1)*5}/{sample_times}")

            current_prompt = e['prompt_en']
            current_video_name = current_prompt + "-" + str((sample_idx+1)*5) + ".mp4"
            current_save_path = os.path.join(args.save_path, current_video_name)

            if os.path.exists(current_save_path):
                print(f"File already exists: {current_save_path}")
                continue
            print(f"Generating video for prompt: {current_prompt}")

            current_prompt = [current_prompt] * 5
            response = model.generate_video_content_batch(
                current_prompt, args.cfg, args.generation_nums,
                cache_percent_list_mlp=args.cache_percent_list_mlp
            )
            save_video_mp4(response, args.save_path, current_prompt[0], sample_idx*5)

            for i_layer in range(len(model.llm.model.layers)):
                model.llm.model.layers[i_layer].self_attn.query_cache = []
                model.llm.model.layers[i_layer].self_attn.cache_if_skip_k_v_projection = []
                model.llm.model.layers[i_layer].cache_attn_input = []
                model.llm.model.layers[i_layer].cache_attn = []
                model.llm.model.layers[i_layer].cache_mlp_input = []
                model.llm.model.layers[i_layer].cache_mlp = []
                model.llm.model.layers[i_layer].cache_if_skip_mlp = []
        
        
    print("end time: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))