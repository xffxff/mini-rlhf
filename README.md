# mini-RLHF (WIP)

A PyTorch implementation of RLHF (Reinforcement Learning from Human Feedback) for educational purposes. This code is based on [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) but excludes the DeepSpeed part.

Key features of this implementation:


- One file for each stage, including supervised fine-tuning (sft), reward model finetune, and RLHF itself. This may lead to some repeated code, but without the need to jump between different files when you want to learn a specific stage.

- Runs on a single GPU, no distributed training. This makes it easier to debug.

- No optimization for speed and memory. 

- No complex command-line interface. Simply execute the respective file, and you will obtain the desired results.
