# FastCar

This repository provides an overview of all resources for the paper ["FastCar: Cache Attentive Replay for
Fast Auto-Regressive Video Generation"]().


FastCar is the first framework that is specially designed for the acceleration of the auto-regressive video generation models.

FastCar additionally takes the temporal redundancy into consideration compared to the previous efficient techniques which focus on spatial redundancy for auto-regressive image generation.

FastCar accelerates the video generation by replaying the feedforward layers with the attentive cache of the previous frame.

FastCar provides the complementary efficient technique for sparse attention based approaches.

## Quick Start

### Model Preparation
Please follow the instruction of environment setup and download the checkpoint from the [VILA-U](https://github.com/mit-han-lab/vila-u). 

Note: I have put the new model.py in "vila_u/train/transformers_replace/models/llama/modeling_llama.py"

### Evaluation
Please follow the instruction from [VBench](https://github.com/Vchitect/VBench) to set up the evaluation.

Or simply install with `pip install vbench`

Then, use the scrips in `scripts-benchmark` to generate videos for VBench evaluation.



