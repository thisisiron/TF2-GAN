# TF2-GAN 
<p align="center">
    <img src="https://pbs.twimg.com/profile_images/1103339571977248768/FtFnqC38_400x400.png" width="200"\>
</p>

## Introduction
This code implements a typical GAN using Tenseflow 2.0. 
I'm following the suggestions in the possible papers, and I have changed the model structure slightly for simple task only. 
However, the main content in the paper has been implemented as it is.

## Requirements
Tensorflow 2.0
Tensorflow Datasets

## How to Run 
```
cd GAN_DIR_YOU_WANT
python train.py
```

## File structure
| Name     | Description                                 |
|----------|---------------------------------------------|
| utils.py | Loss function, Image storage function, etc. |
| model.py | Model Architecture                          |
| train.py | Model learning and Loading datasets         |

## GAN List
- [X] GAN
- [ ] CGAN
- [ ] ACGAN
- [ ] infoGAN
- [ ] BEGAN
- [ ] Stacked GAN
- [ ] SRGAN
- [ ] EBGAN
- [X] DCGAN
- [ ] SAGAN
- [ ] LSGAN
- [ ] CycleGAN
- [X] WGAN
- [X] WGAN-GP 
