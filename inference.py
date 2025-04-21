import argparse
import cv2
import numpy as np
import os
import random
from einops import rearrange
import torchvision
import imageio
import ast
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
def save_video_mp4(response, path, fps=8):
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
        video_path = os.path.join(path, f"video_{i}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
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
    parser.add_argument("--cache_percent_list_mlp", type=parse_list, default=[None]*32)
    args = parser.parse_args()

    if args.model_path is not None:
        model = vila_u.load(args.model_path)
    else:
        raise ValueError("No model path provided!")

    if args.query is not None:
        generation_config = model.default_generation_config
        generation_config.temperature = args.temperature
        generation_config.top_p = args.top_p
        if args.image_path is not None:
            image = vila_u.Image(args.image_path)
            response = model.generate_content([image, args.query])
            print("\033[1;32mResponse:\033[0m", response)
            exit()
        elif args.video_path is not None:
            video = vila_u.Video(args.video_path)
            response = model.generate_content([video, args.query])
            print("\033[1;32mResponse:\033[0m", response)
            exit()
        else:
            raise ValueError("No visual content input!")
    elif args.prompt is not None:
        if args.video_generation:
            # response = model.generate_video_content(args.prompt, args.cfg, args.generation_nums)

            cache_percent_list_mlp = args.cache_percent_list_mlp

            print("=========================================")
            print("cache_percent_list_mlp: ", cache_percent_list_mlp)
            print("=========================================")
            
            response = model.generate_video_content_batch(
                [args.prompt] * 5,
                args.cfg, 
                args.generation_nums,
                cache_percent_list_mlp=cache_percent_list_mlp
            )

            save_video(response, args.save_path)
            save_video_mp4(response, args.save_path)

            total_head = 0
            skipped_head = 0
            total_attn = 0
            skipped_attn = 0
            total_mlp = 0
            skipped_mlp = 0
            for i_layer in range(len(model.llm.model.layers)):
                for e in model.llm.model.layers[i_layer].self_attn.cache_if_skip_k_v_projection:
                    total_head += (e.shape[0] * e.shape[1])
                    skipped_head += torch.sum(e).item()
                    # total_attn += 1
                    # if torch.all(e).item():
                    #     skipped_attn += 1
                for e in model.llm.model.layers[i_layer].cache_if_skip_mlp:
                    if e.shape[0] != 2 and e.shape[1] != 32 and e.shape[2] != 1:
                        import pdb; pdb.set_trace()
                    if len(e.shape) > 2:
                        # head level below
                        total_mlp += e.shape[0] * e.shape[1]
                        skipped_mlp += torch.sum(e).item()
                    else:
                        total_mlp += 1
                        if torch.all(e).item():
                            skipped_mlp += 1
            if total_head > 0:
                print(f"Skipped Head: {skipped_head}/{total_head}")
                print(f"Skipped Head Ratio: {100*skipped_head/total_head:.2f}%")
            if total_attn > 0:
                print(f"Skipped Attention: {skipped_attn}/{total_attn}")
                print(f"Skipped Attention Ratio: {100*skipped_attn/total_attn:.2f}%")
            if total_mlp > 0:
                print(f"Skipped MLP: {skipped_mlp}/{total_mlp}")
                print(f"Skipped MLP Ratio: {100*skipped_mlp/total_mlp:.2f}%")
            exit()
        else:
            response = model.generate_image_content(args.prompt, args.cfg, args.generation_nums)
            save_image(response, args.save_path)
            exit()
    else:
        raise ValueError("No query or prompt provided!")