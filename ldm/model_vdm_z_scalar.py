# Copyright 2022 The VDM Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# sbatch --gres=gpu:4 --partition=kuleshov --exclude=compling-compute-02,rush-compute-02  run.sh -m ldm.main --mode train --config=ldm/configs/cifar10-small-discrete-conditioned.py --workdir /share/kuleshov/ssahoo/diffusion_models/latent_z/ --config.vdm_type=z_scalar --config.model.reparam_type=true --config.model.encoder=unet  --config.model.sm_n_timesteps=0

from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from ldm import model_vdm
from ldm import model_vdm_conditioned


class UnetEncoderGaussian(nn.Module):
  config: model_vdm.VDMConfig

  @nn.compact
  def __call__(self, z, deterministic):
    conditioning=jnp.zeros((z.shape[0],), dtype='uint8')
    t = jnp.zeros((z.shape[0],), z.dtype)
    config = self.config
    # Compute conditioning vector based on 'g_t' and 'conditioning'
    n_embd = self.config.sm_n_embd
  
    temb = model_vdm.get_timestep_embedding(t, n_embd)
    cond = jnp.concatenate([temb, conditioning[:, None]], axis=1)
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense0')(cond))
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense1')(cond))

    # Concatenate Fourier features to input
    if config.with_fourier_features:
      z_f = model_vdm.Base2FourierFeatures(start=6, stop=8, step=1)(z)
      h = jnp.concatenate([z, z_f], axis=-1)
    else:
      h = z

    # Linear projection of input
    h = nn.Conv(features=n_embd, kernel_size=(
        3, 3), strides=(1, 1), name='conv_in')(h)
    hs = [h]

    # Downsampling
    for i_block in range(self.config.forward_n_layer):
      block = model_vdm.ResnetBlock(config, out_ch=n_embd, name=f'down.block_{i_block}')
      h = block(hs[-1], cond, deterministic)[0]
      if config.with_attention:
        h = model_vdm.AttnBlock(num_heads=1, name=f'down.attn_{i_block}')(h)
      hs.append(h)

    # Middle
    h = hs[-1]
    h = model_vdm.ResnetBlock(config, name='mid.block_1')(h, cond, deterministic)[0]
    h = model_vdm.AttnBlock(num_heads=1, name='mid.attn_1')(h)
    h = model_vdm.ResnetBlock(config, name='mid.block_2')(h, cond, deterministic)[0]

    # Predict noise
    normalize = nn.normalization.GroupNorm()
    h = nn.swish(normalize(h))
    h = nn.Conv(
        features=1,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv_out')(h)
    h = nn.swish(h.reshape((h.shape[0], -1)))
    mu = nn.Dense(self.config.latent_size,
                  name='dense_layer_final_mu')(h)
    sigma = nn.Dense(self.config.latent_size,
                  name='dense_layer_final_sigma')(h)
    return mu, jax.nn.softplus(sigma)


class VDM(nn.Module):
  config: model_vdm.VDMConfig

  def setup(self):
    self.encdec = model_vdm.EncDec(self.config)
    self.score_model = model_vdm.ScoreUNet(self.config)
    if self.config.latent_type in {'gumbel', 'topk'}:
      self.encoder_model = model_vdm_conditioned.ENCODER_MODELS[self.config.encoder](config=self.config)
    elif self.config.latent_type == 'gaussian':
      self.encoder_model = UnetEncoderGaussian(self.config)
    self.condition = self.config.condition
    if self.config.gamma_type == 'learnable_nnet':
      self.gamma = model_vdm_conditioned.NoiseSchedule_NNet(self.config)
    elif self.config.gamma_type == 'film':
      self.gamma = model_vdm_conditioned.NoiseSchedule_Film(self.config)
    elif self.config.gamma_type == 'poly_fixedend':
      self.gamma = model_vdm_conditioned.NoiseSchedule_polynomial(self.config)
    self.epsilon = self.config.epsilon

  def apply_gamma(self, t):
    if jnp.isscalar(t) or len(t.shape) == 0:
      batch_size = 1
    else:
      batch_size = t.shape[0]
    return self._get_gamma(
      jnp.zeros((batch_size, self.config.latent_size)), t)

  def _get_gumbel_embedding(self, logits, deterministic, tau=1.0):
    assert len(logits.shape) == 2
    if deterministic: # eval
      gumbel_noise = 0.
    else: # train
      gumbel_noise = jax.random.gumbel(self.make_rng('sample'), logits.shape)
    logits = (logits + gumbel_noise) / tau
    soft_argmax = jax.nn.softmax(logits)
    hard_argmax = jax.nn.one_hot(
      jnp.argmax(logits, axis=-1), self.config.latent_size)
    assert soft_argmax.shape == hard_argmax.shape == logits.shape
    return jax.lax.stop_gradient(hard_argmax - soft_argmax) + soft_argmax

  def _gumbel_kl_loss(self, logits):
    q_z = jax.nn.softmax(logits)
    log_q_z = jax.nn.log_softmax(logits)
    return jnp.sum(
      q_z * (log_q_z - jnp.log(1.0 / self.config.latent_size)),
      axis=1)

  def _gumbel_embedding_and_loss(self, orig_f, step, deterministic):
    logits = self.encoder_model(orig_f,  deterministic)
    embedding = self._get_gumbel_embedding(
      logits,
      deterministic,
      # tau varies from 1 -> 0.1
      tau=jnp.maximum(0.5, jnp.exp(- 0.00001 * step)))
    return embedding, self._gumbel_kl_loss(logits)

  def _gamma_noise(self, k, shape, gamma_tau=10.0):
    noise = jax.random.gamma(
        self.make_rng('sample'), 1.0 / k, shape=(10, * shape))
    beta = k / jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    beta = beta[:, None, None]
    assert len(beta.shape) == len(noise.shape)
    s = noise / beta
    s = jnp.sum(s, axis=0)
    s = s - jnp.log(10.0)
    s = gamma_tau * (s / k)
    return s


  def _topk_embedding_and_loss(self, orig_f, k, deterministic):
    logits = self.encoder_model(orig_f,  deterministic)
    if deterministic: # eval
      gamma_noise = 0.
    else: # train
      gamma_noise = self._gamma_noise(k=k, shape=logits.shape)
    kl_loss = self._gumbel_kl_loss(logits)
    logits = logits + gamma_noise
    
    # sahoo et al. https://arxiv.org/abs/2205.15213
    # code: https://github.com/martius-lab/solver-differentiation-identity/blob/main/discrete-VAE-experiments-neurips-identity.ipynb
    logits = logits - jnp.mean(logits, axis=1, keepdims=True)
    soft_topk = logits / jnp.linalg.norm(logits, axis=1, keepdims=True)
    
    top_k_vals, _ = jax.lax.top_k(logits, k)
    assert top_k_vals.shape == (logits.shape[0], k)
    hard_topk = (logits >= top_k_vals[:, -1][:, None]).astype(float)
    # jax.debug.print('logits {x}', x=logits)
    # jax.debug.print('top_k_vals {x}', x=top_k_vals)
    # jax.debug.print('hard_topk {x}', x=jnp.sum(hard_topk, axis=1))
    embedding = jax.lax.stop_gradient(hard_topk - soft_topk) + soft_topk
    return embedding, kl_loss

  def _get_gamma(self, embedding, t):
    _, nonlinear = self.gamma(embedding, t)
    return jnp.squeeze(nonlinear)[:, None, None, None]

  def __call__(
      self, images, labels, conditioning, step, deterministic: bool = True):
    
    x = images
    n_batch = images.shape[0]

    # 3. DIFFUSION LOSS
    # sample time steps
    rng1 = self.make_rng('sample')
    if self.config.antithetic_time_sampling:
      t0 = jax.random.uniform(rng1)
      t = jnp.mod(t0 + jnp.arange(0., 1., step=1. / n_batch), 1.)
    else:
      t = jax.random.uniform(rng1, shape=(n_batch,))

    # discretize time steps if we're working with discrete time
    T = self.config.sm_n_timesteps
    if T > 0:
      t = jnp.ceil(t * T) / T

    # Process input
    orig_f = self.encdec.encode(x)
    if self.config.reparam_type == 'true':
      if self.config.latent_type == 'gumbel':
        embedding, kl_z = self._gumbel_embedding_and_loss(
          orig_f=orig_f, step=step, deterministic=deterministic)
      elif self.config.latent_type == 'topk':
        embedding, kl_z = self._topk_embedding_and_loss(
          orig_f=orig_f, k=15, deterministic=deterministic)
      elif self.config.latent_type == 'gaussian':
        mu_z, var_z = self.encoder_model(orig_f,  deterministic)
        eps_z = jax.random.normal(self.make_rng('sample'), shape=mu_z.shape)
        embedding = mu_z + jnp.sqrt(var_z) * eps_z
        kl_z = 0.5 * jnp.sum(
          mu_z ** 2 + var_z - jnp.log(var_z) - 1.,
          axis=1)
    else:
      embedding = jax.nn.one_hot(labels, 10)
      kl_z = 0.0
    # jax.debug.print('kl_z {x}', x=kl_z)
    # jax.debug.print('logits {x}', x=logits)
    # jax.debug.print('embedding {x}', x=embedding)
    g_0 = self._get_gamma(embedding, jnp.zeros_like(t))
    g_1 = self._get_gamma(embedding, jnp.ones_like(t))
    g_t = self._get_gamma(embedding, t)
    var_t = nn.sigmoid(g_t)
    var_0 = nn.sigmoid(g_0)
    var_1 = nn.sigmoid(g_1)
    # 1. Reconstruction loss
    eps_0 = jax.random.normal(self.make_rng('sample'), shape=orig_f.shape)
    # z_0 = jnp.sqrt(1. - var_0) * f_0 + jnp.sqrt(var_0) * eps_0
    z_0_rescaled = orig_f + jnp.exp(0.5 * g_0) * eps_0  # = z_0 / sqrt(1 - var)
    loss_recon = - self.encdec.logprob(x, z_0_rescaled, g_0)

    # 2. Latent Loss
    # KL z1 with N(0,1) prior
    mean1_sqr = (1. - var_1) * jnp.square(orig_f)
    loss_klz = 0.5 * jnp.sum(
      mean1_sqr + var_1 - jnp.log(var_1) - 1.,
      axis=(1, 2, 3))
  
    eps = jax.random.normal(self.make_rng('sample'), shape=orig_f.shape)
    z_t = jnp.sqrt(1. - var_t) * orig_f + jnp.sqrt(var_t) * eps
    if self.config.z_conditioning:
      conditioning = embedding
    else:
      conditioning = conditioning[:, None]

    eps_hat = self.score_model(
      z_t, jnp.squeeze(g_t), conditioning, deterministic)
    if T == 0:
      _, g_t_grad = jax.jvp(
        self._get_gamma,
        (embedding, t),
        (jnp.zeros_like(embedding), jnp.ones_like(t)))
      # jax.debug.print('g_t_grad {x}', x=g_t_grad)
      # jax.debug.print('t {x}', x=t)
      assert jnp.squeeze(g_t_grad).shape == t.shape
      loss_diff = .5 * jnp.sum(
        g_t_grad * jnp.square(eps - eps_hat),
        axis=[1, 2, 3])
    else:
      # loss for finite depth T, i.e. discrete time
      s = t - (1./T)
      g_s = self._get_gamma(embedding, s)
      # jax.debug.print('g_s.shape {x}', x=g_s.shape)
      # jax.debug.print('g_t.shape {x}', x=g_t.shape)

      assert g_s.shape == g_t.shape
      loss_diff = .5 * T * jnp.sum(
        jnp.expm1(g_t - g_s) * jnp.square(eps - eps_hat),
        axis=[1, 2, 3])
  
    return model_vdm.VDMOutput(
        loss_recon=loss_recon,
        loss_klz=kl_z + loss_klz,
        loss_diff=loss_diff,
        var_0=jnp.mean(var_0),
        var_1=jnp.mean(var_1),
    )

  def _get_deterministic_embedding(self, batch_size):
    k = self.config.latent_size
    if self.config.latent_type == 'gumbel':
      return jax.nn.one_hot(jnp.ones(batch_size), k)
    elif self.config.latent_type == 'topk':
      # TODO: configure this for any `k` in topk
      ones = jnp.ones((batch_size, 15))
      zeros = jnp.zeros((batch_size, k - 15))
      return jnp.concatenate([ones, zeros], axis=1)
    elif self.config.latent_type == 'gaussian':
      return jnp.zeros((batch_size, k))

  def sample(self, i, T, z_t, conditioning, rng): 
    rng_body = jax.random.fold_in(rng, i)
    eps = jax.random.normal(rng_body, z_t.shape)

    t = (T - i) / T
    s = (T - i - 1) / T
    
    embedding = self._get_deterministic_embedding(z_t.shape[0])
    
    g_t = self._get_gamma(embedding, t)
    g_s = self._get_gamma(embedding, s)

    if self.config.z_conditioning:
      conditioning = embedding

    eps_hat = self.score_model(
        z_t,
        jnp.squeeze(g_t) * jnp.ones((z_t.shape[0],), z_t.dtype),
        conditioning,
        deterministic=True)
  
    a = nn.sigmoid(-g_s)
    b = nn.sigmoid(-g_t)
    c = - jnp.expm1(g_s - g_t)
    sigma_t = jnp.sqrt(nn.sigmoid(g_t))
    z_s_mean = jnp.sqrt(a / b) * (z_t - sigma_t * c * eps_hat) 

    return z_s_mean + jnp.sqrt((1. - a) * c) * eps

  def generate_x(self, z_0):
    g_0 = self._get_gamma(
      self._get_deterministic_embedding(z_0.shape[0]),
      jnp.zeros((z_0.shape[0],)))
    var_0 = nn.sigmoid(g_0)
    z_0_rescaled = z_0 / jnp.sqrt(1. - var_0)

    logits = self.encdec.decode(z_0_rescaled, g_0)

    # get output samples
    if self.config.sample_softmax:
      out_rng = self.make_rng('sample')
      samples = jax.random.categorical(out_rng, logits)
    else:
      samples = jnp.argmax(logits, axis=-1)

    return samples