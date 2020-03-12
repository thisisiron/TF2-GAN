# TF2-GAN 
<p align="center">
    <img src="https://pbs.twimg.com/profile_images/1103339571977248768/FtFnqC38_400x400.png" width="200"\>
</p>

## Introduction
Typical GANs are implemented as Tensorflow 2. <br>
I followed the suggestions in the papers, and I slightly changed the model structure or optimizer for simple task. <br>

## Requirements
Tensorflow 2.0<br>
Tensorflow Datasets<br>
Tensorflow-addons

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
- [X] CGAN
- [X] DCGAN
- [X] LSGAN
- [X] WGAN
- [X] WGAN-GP 
- [X] CycleGAN
- [X] StarGAN
- [ ] SRGAN
- [ ] SAGAN
- [ ] ACGAN
- [ ] infoGAN
- [ ] BEGAN
- [ ] BigGAN
- [ ] Stacked GAN
- [ ] EBGAN
