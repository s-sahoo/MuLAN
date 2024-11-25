# [Diffusion Models With Learned Adaptive Noise (NeurIPS 2024, spotlight)](https://arxiv.org/abs/2312.13236)
By [Subham Sekhar Sahoo](https://s-sahoo.github.io), [Aaron Gokaslan](https://skylion007.github.io), [Chris De Sa](https://www.cs.cornell.edu/~cdesa/), [Volodymyr Kuleshov](https://www.cs.cornell.edu/~kuleshov/)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusion-models-with-learned-adaptive-noise/density-estimation-on-imagenet-32x32-1)](https://paperswithcode.com/sota/density-estimation-on-imagenet-32x32-1?p=diffusion-models-with-learned-adaptive-noise)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/diffusion-models-with-learned-adaptive-noise/density-estimation-on-cifar-10)](https://paperswithcode.com/sota/density-estimation-on-cifar-10?p=diffusion-models-with-learned-adaptive-noise)

We introduce *MuLAN* (MUltivariate Learned Adaptive Noise) that learns the forward noising process from the data. In this work we dispel the widely held assumption that the ELBO is invariant to the noise process. Empirically, MULAN sets a new **state-of-the-art** in density estimation on CIFAR-10 and ImageNet and reduces the number of training steps by 50% as summarized in the table below (Likelihood in bits per dimension):
|                         | CIFAR-10 $(\downarrow)$ | ImageNet $(\downarrow)$ |
|----------------------------|-------------|------------|
| PixelCNN                   | 3.03        | 3.83       |
| Image Transformer          | 2.90        | 3.77       |
| DDPM                       | $\leq$ 3.69 | /          |
| ScoreFlow                  | 2.83        | 3.76       |
| VDM                        | $\leq$ 2.65 | $\leq$ 3.72|
| Flow Matching              | 2.99        | /          |
| Reflected Diffusion Models | 2.68        | 3.74       |
| **MuLAN** (Ours)           | **2.55**    | **3.67**   |

Note:  We only compare with results achieved without data augmentation.

## Setup
### Dependencies
Install the dependencies via `pip` using the following commands:
```
pip install -U "jax[cuda12_pip]<=0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install -r requirements.txt
```

### Dataset
The experiments were conducted on CIFAR-10 and ImageNet32 datasets. We used the dataloader provided by `tensorflow_datasets`. To maintain consistency with previous baselines, we utilized the [older-version](http://image-net.org/small/train_32x32.tar) of ImageNet32, which is no longer publicly available. Therefore, we provide the dataset, which can be downloaded from this [google-drive link](https://drive.google.com/file/d/1I-QvjLRa1kVxc3iX05pmDEKEM0JeAq46/view?usp=share_link). To use this dataset please download the tar file and extract it into the `~/tensorflow_datasets` directory. The final structure should look like the following:
```
~/tensorflow_datasets/downsampled_imagenet/32x32/2.0.0/downsampled_imagenet-train.tfrecord-000*-of-00032
```

### Code
The implementation of MuLAN can be found in [ldm/model_mulan_epsilon.py](ldm/model_mulan_epsilon.py). The denoising model uses `noise-parameterization`, as described in `suppl. 11.1.1` of the paper. The file [ldm/model_mulan_velocity.py](ldm/model_mulan_velocity.py) implements velocity parameterization, as detailed in `suppl. 11.1.2` of the paper. 


## Likelihood Estimation
Download the **checkpoints and Tensorboard logs** from the [Google Drive](https://drive.google.com/drive/folders/1RVnTljGDj4G8gu2ltYFX0wwD9OlKRpWT?usp=sharing) folder. Please note that the eval likelihood / BPD (bits per dimension) in the tensorboard log was computed using a partial dataset, which is why they are worse than the numbers reported in the paper. To compute the BPD accurately, use the following commands:

### Exact likelihood Estimation
To compute the exact likelihood as per `suppl. 15.2` use the following commands:
```
JAX_DEFAULT_MATMUL_PRECISION=float32 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 python -m ldm.eval_bpd --config=ldm/configs/cifar10-conditioned.py --config.vdm_type=mulan_velocity  --checkpoint_directory=/share/kuleshov/ssahoo/diffusion_models/velocity_parameterization/1124188-vdm_type=mulan_velocity-topk_noise_type=gamma-ckpt_restore_dir/checkpoints-0 --checkpoint=223

JAX_DEFAULT_MATMUL_PRECISION=float32 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 python -m ldm.eval_bpd --config=ldm/configs/imagenet32.py --config.vdm_type=mulan_velocity  --config.model.velocity_from_epsilon=True --checkpoint_directory=/share/kuleshov/ssahoo/diffusion_models/imagenet_mulan_epsilon/checkpoints-0 --checkpoint=220
```
The code for exact likelihood estimation supports multi-gpu evaluations.

### Variance Lower Bound Estimation
To compute the likelihood using the Variance Lower Bound (VLB) as per `suppl. 15.1` in the paper, use the following commands:
```
JAX_DEFAULT_MATMUL_PRECISION=float32 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 python -m ldm.eval_bpd --config=ldm/configs/cifar10-conditioned.py  --checkpoint_directory=/path/to/checkpoints/cifar10 --checkpoint=223 --bpd_eval_method=dense --config.training.batch_size_eval=16

JAX_DEFAULT_MATMUL_PRECISION=float32 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 python -m ldm.eval_bpd --config=ldm/configs/imagenet32.py --config.vdm_type=mulan_velocity  --config.model.velocity_from_epsilon=True --checkpoint_directory=/path/to/checkpoints/imagenet --checkpoint=200 --bpd_eval_method=dense --config.training.batch_size_eval=16
```
The code for VLB estimation **doesn't** support multi-gpu evaluations and hence must be run on a single gpu.

## Training
For `CIFAR-10`, we trained our models on `V100s` using the following `slurm` commands:
```
sbatch -J cifar --partition=kuleshov --gres=gpu:4 run.sh -m ldm.main --mode train --config=ldm/configs/cifar10-conditioned.py --workdir /path/to/experiment_dir
```

For `ImageNet-32`, we trained our models on `A100s` using the following command:
```
sbatch -J img --partition=gpu --gres=gpu:4 run.sh -m ldm.main --mode train --config=ldm/configs/imagenet32.py --workdir /path/to/experiment_dir
```

### Acknowledgements
This repository was built off of [VDM](https://github.com/google-research/vdm).


## Citation
```bib
@inproceedings{
sahoo2024diffusion,
title={Diffusion Models With Learned Adaptive Noise},
author={Subham Sekhar Sahoo and Aaron Gokaslan and Christopher De Sa and Volodymyr Kuleshov},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=loMa99A4p8}
}
```
