# [Diffusion Models With Learned Adaptive Noise](https://arxiv.org/abs/2312.13236)
By [Subham Sekhar Sahoo](https://s-sahoo.github.io), [Aaron Gokaslan](https://skylion007.github.io), [Chris De Sa](https://www.cs.cornell.edu/~cdesa/), [Volodymyr Kuleshov](https://www.cs.cornell.edu/~kuleshov/)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusion-models-with-learned-adaptive-noise/density-estimation-on-imagenet-32x32-1)](https://paperswithcode.com/sota/density-estimation-on-imagenet-32x32-1?p=diffusion-models-with-learned-adaptive-noise)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusion-models-with-learned-adaptive-noise/density-estimation-on-cifar-10)](https://paperswithcode.com/sota/density-estimation-on-cifar-10?p=diffusion-models-with-learned-adaptive-noise)

We introduce *MuLAN* (MUltivariate Learned Adaptive Noise) that learns the forward noising process from the data. In this work we dispel the widely held assumption that the ELBO is invariant to the noise process. Empirically, MULAN sets a new **state-of-the-art** in density estimation on CIFAR-10 and ImageNet and reduces the number of training steps by 50% as summarized in the table below (Likelihood in bits per dimension (BPD)):
|                         | CIFAR-10 $(\downarrow)$ | ImageNet $(\downarrow)$ |
|----------------------------|-------------|-----------|
| PixelCNN                   | 3.03        | 3.83      |
| Image Transformer          | 2.90        | 3.77      |
| DDPM                       | $\leq$ 3.69 | /         |
| ScoreFlow                  | 2.83        | 3.76      |
| VDM                        | $\leq$ 2.65 | $\leq$ 3.72|
| Flow Matching              | 2.99        | /         |
| Reflected Diffusion Models | 2.68        | 3.74      |
| **MuLAN** (Ours)           | **2.55**    | **3.67**  |

Note:  We only compare with results achieved without data augmentation.

## Checkpoints and Tensorboard logs
Download the checkpoints and logs from the [Google Drive](https://drive.google.com/drive/folders/1RVnTljGDj4G8gu2ltYFX0wwD9OlKRpWT?usp=sharing) folder. Please note that the eval BPD (bits per dimension) in the tensorboard log was computed using a partial dataset, which is why they are worse than the numbers reported in the paper. To compute BPD accurately, use the following `slurm` commands:

### Exact likelihood Estimation
To compute the exact likelihood as per `suppl. 15.2` use the following command:
```
sbatch -J cifar_eval --partition=kuleshov --gres=gpu:4 run.sh -m ldm.eval_bpd --config=ldm/configs/cifar10-conditioned.py --config.vdm_type=z_pp_velocity  --checkpoint_directory=/share/kuleshov/ssahoo/diffusion_models/velocity_parameterization/1124188-vdm_type=z_pp_velocity-topk_noise_type=gamma-ckpt_restore_dir/checkpoints-0 --checkpoint=223

sbatch -J img_eval --partition=gpu --gres=gpu:4 run.sh -m ldm.eval_bpd --config=ldm/configs/imagenet32.py --config.vdm_type=z_pp_velocity  --config.model.velocity_from_epsilon=True --checkpoint_directory=/share/kuleshov/ssahoo/diffusion_models/imagenet_mulan_epsilon/checkpoints-0 --checkpoint=220
```

### Variance Lower Bound Estimation
To compute the likelihood using the Variance Lower Bound (VLB) as per `suppl. 15.1` in the paper, use the following command:
```
sbatch -J cifar_eval_dense --partition=kuleshov --gres=gpu:1 run.sh -m ldm.eval_bpd --config=ldm/configs/cifar10-conditioned.py --config.vdm_type=z_pp_velocity  --checkpoint_directory=/share/kuleshov/ssahoo/diffusion_models/opensource_checkpoints/cifar10 --checkpoint=223 --bpd_eval_method=dense --config.training.batch_size_eval=16

sbatch -J img_eval --partition=gpu --gres=gpu:1 run.sh -m ldm.eval_bpd --config=ldm/configs/imagenet32.py --config.vdm_type=z_pp_velocity  --config.model.velocity_from_epsilon=True --checkpoint_directory=/share/kuleshov/ssahoo/diffusion_models/opensource_checkpoints/imagenet --checkpoint=200 --bpd_eval_method=dense --config.training.batch_size_eval=16
```

## Training from scratch
For `CIFAR-10`, we trained our models on `V100s` using the following command:
```
sbatch -J cifar --partition=kuleshov --gres=gpu:4 run.sh -m ldm.main --mode train --config=ldm/configs/cifar10-conditioned.py --workdir /share/kuleshov/ssahoo/diffusion_models/reproduce --config.vdm_type=z_pp_velocity
```

For `ImageNet-32`, we trained our models on `A100s` using the following command:
```
sbatch -J img --partition=gpu --gres=gpu:4 run.sh -m ldm.main --mode train --config=ldm/configs/imagenet32.py --workdir /share/kuleshov/ssahoo/diffusion_models/reproduce
```

### Acknowledgements
This repository was built off of [VDM](https://github.com/google-research/vdm).


## Citation
```bib
@article{sahoo2023diffusion,
  title={Diffusion Models With Learned Adaptive Noise},
  author={Sahoo, Subham Sekhar and Gokaslan, Aaron and De Sa, Chris and Kuleshov, Volodymyr},
  journal={arXiv preprint arXiv:2312.13236},
  year={2023}
}
```
