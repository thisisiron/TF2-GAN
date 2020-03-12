# Star GAN 

## Model Architecture
![](./images/model.png)

## Loss Function
### Adverarial Loss
![](./images/adversarial_loss.png)

### Reconstruction Loss 
![](./images/reconstruction_loss.png)

### Domain Classification Loss 
![](./images/domain_classification_loss_real.png)
![](./images/domain_classification_loss_fake.png)

### Full Objective
![](./images/full_objective.png)

## Result
| Input | Blond Hair || Input | Black Hair || Input | Brown_Hair || Input | Aged || Input | Gender |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  ![](./images/blonde_ori1.png) | ![](./images/blonde_tar1.png) || ![](./images/black_ori1.png) | ![](./images/black_tar1.png) || ![](./images/brown_ori1.png) | ![](./images/brown_tar1.png) || ![](./images/aged_ori1.png) | ![](./images/aged_tar1.png) || ![](./images/gender_ori1.png) | ![](./images/gender_tar1.png) |
| ![](./images/blonde_ori2.png) | ![](./images/blonde_tar2.png) || ![](./images/black_ori2.png) | ![](./images/black_tar2.png) || ![](./images/brown_ori2.png) | ![](./images/brown_tar2.png) || ![](./images/aged_ori2.png) | ![](./images/aged_tar2.png) || ![](./images/gender_ori2.png) | ![](./images/gender_tar2.png) |
| ![](./images/blonde_ori3.png) | ![](./images/blonde_tar3.png) || ![](./images/black_ori3.png) | ![](./images/black_tar3.png) || ![](./images/brown_ori3.png) | ![](./images/brown_tar3.png) || ![](./images/aged_ori3.png) | ![](./images/aged_tar3.png) || ![](./images/gender_ori3.png) | ![](./images/gender_tar3.png) |


## Dataset
```
./download_dataset monet2photo
```

## Reference
[StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/pdf/1711.09020.pdf)
