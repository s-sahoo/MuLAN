# Copyright 2022 The VDM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Get configuration
import ml_collections


def d(**kwargs):
  """Helper of creating a config dict."""
  return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
  """Get the hyperparameters for the model"""
  config = ml_collections.ConfigDict()
  config.exp_name = 'exp_vdm'
  config.model_type = 'model_vdm'
  config.ckpt_restore_dir = 'None'

  config.data = d(
      dataset='cifar10',  # cifar10/cifar10_aug
      ignore_cache=False,
  )
  
  config.vdm_type = 'mulan_velocity' # mulan_epsilon, mulan_velocity

  config.model = d(
      unet_type='vdm', # ldm, vdm
      vocab_size=256,
      sample_softmax=False,
      antithetic_time_sampling=True,
      with_fourier_features=True,
      with_attention=False,
    
      condition='input', # label, input, ignore
      encoder='unet', # cnn, unet
      forward_n_layer=4,
      latent_size=50,
      latent_k=15,
      model_time=False,
      epsilon=0.0,
      # dense_monotone, dense_monotone_softplus
      monotone_layer='dense_monotone',
      gamma_type='poly_fixedend', # poly_fixedend_v2, poly_fixedend, filmv2
      # learnable_nnet_v2, learnable_nnet, film, learnable_nnet_time_matrix,
      # learnable_nnet_linear, poly_fixedend_stop_grad
      latent_type='topk', # gumbel, topk, gaussian
      z_conditioning=True,
      trace_matching=False,
      importance_sampling=False,
      topk_noise_type='gamma', # gamma, gumbel
      sigma_prior=1.0,
      reparam_type='true', # mu_sigma, noise, input, true
      # configurations of the noise schedule
      gamma_min=-13.3,
      gamma_max=5.,
      velocity_from_epsilon=False,

      # blur schedule
      sigma_type='no_blur',
      sigma_min=0.0,
      sigma_max=20.0,

      # configurations of the score model
      sm_n_timesteps=0,  # 1000 for T = 1000
      sm_n_embd=128,
      sm_n_layer=32,
      sm_pdrop=0.1,
  )

  config.training = d(
      seed=1,
      substeps=1000,
      num_steps_lr_warmup=100,
      num_steps_train=10_000_000,
      num_steps_eval=100,
      batch_size_train=128,
      batch_size_eval=128,
      steps_per_logging=1000,
      steps_per_eval=10_000,
      steps_per_save=10_000,
      profile=False,
  )

  config.optimizer = d(
      name='adamw',
      args=d(
          b1=0.9,
          b2=0.99,
          eps=1e-8,
          weight_decay=0.01,
      ),
      learning_rate=2e-4,
      lr_decay=False,
      ema_rate=0.9999,
  )

  return config
