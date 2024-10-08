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
# sbatch --gres=gpu:4 --exclude=compling-compute-02,rush-compute-02  run.sh -m ldm.main --mode train --config=ldm/configs/cifar10-small-discrete-conditioned.py --workdir /share/kuleshov/ssahoo/diffusion_models/latent_z/ --config.vdm_type=z_pp --config.model.reparam_type=true --config.model.encoder=unet  --config.model.sm_n_timesteps=0 --config.model.gamma_type=learnable_nnet_time_matrix

from flax import linen as nn
import jax
from jax import numpy as jnp

from ldm import model_vdm
from ldm import ldm_unet


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


class CNN(nn.Module):
  config: model_vdm.VDMConfig

  def setup(self):
    self.conv1 = nn.Conv(features=32, kernel_size=(3,3), padding='SAME', name='CONV1')
    self.conv2 = nn.Conv(features=16, kernel_size=(3,3), padding='SAME', name='CONV2')
    self.linear1 = nn.Dense(self.config.latent_size, name='DENSE')

  def __call__(self, inputs, deterministic=True):
    x = nn.relu(self.conv1(inputs))
    x = nn.relu(self.conv2(x))

    x = x.reshape((x.shape[0], -1))
    x = self.linear1(x)

    return  x


class UnetEncoder(nn.Module):
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
    return nn.Dense(self.config.latent_size,
                    name='dense_layer_final')(h)


ENCODER_MODELS = {'cnn': CNN, 'unet': UnetEncoder}


class VDM(nn.Module):
  config: model_vdm.VDMConfig

  def setup(self):
    self.encdec = model_vdm.EncDec(self.config)
    if self.config.unet_type == 'ldm':
      self.score_model = ldm_unet.UNet(self.config)
    elif self.config.unet_type == 'vdm':
      self.score_model = model_vdm.ScoreUNet(self.config)
    self.condition = self.config.condition
    if self.config.latent_type in {'gumbel', 'topk'}:
      self.encoder_model = ENCODER_MODELS[self.config.encoder](config=self.config)
    elif self.config.latent_type == 'gaussian':
      self.encoder_model = UnetEncoderGaussian(self.config)
    self.gamma = GAMMA_NETWORKS[self.config.gamma_type](self.config)
    self.epsilon = self.config.epsilon
    self.topk_noise_type = self.config.topk_noise_type

  def apply_encoder(self, images_int):
    images = self.encdec.encode(images_int)
    return self.encoder_model(images, deterministic=True)

  def apply_gamma(self, t, x_zero=None, step=0, deterministic : bool = False):
    if jnp.isscalar(t) or len(t.shape) == 0:
      batch_size = 1
    else:
      batch_size = t.shape[0]
    if x_zero is None:
      embedding = jnp.zeros((batch_size, self.config.latent_size))
    else:
      x_zero = self.encdec.encode(x_zero)
      embedding, _ = self._get_embedding_and_kl_z(
        x_zero, step=step, deterministic=deterministic)
    return self._get_gamma(embedding, t)

  def _get_gumbel_embedding(self, logits, deterministic, tau=1.0):
    assert len(logits.shape) == 2
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
      # tau varies from 1 -> 0.5
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
    kl_loss = self._gumbel_kl_loss(logits)
    if self.topk_noise_type == 'gamma':
      logits = logits + self._gamma_noise(k=k, shape=logits.shape)
    elif self.topk_noise_type == 'gumbel':
      logits = logits + jax.random.gumbel(self.make_rng('sample'), logits.shape)
    
    # sahoo et al. https://arxiv.org/abs/2205.15213
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
    return self.gamma(embedding, t)

  def _get_embedding_and_kl_z(self, orig_f, step, deterministic):
    if self.config.latent_type == 'gumbel':
      embedding, kl_z = self._gumbel_embedding_and_loss(
        orig_f=orig_f, step=step, deterministic=deterministic)
    elif self.config.latent_type == 'topk':
      embedding, kl_z = self._topk_embedding_and_loss(
        orig_f=orig_f, k=self.config.latent_k, deterministic=deterministic)
    elif self.config.latent_type == 'gaussian':
      mu_z, var_z = self.encoder_model(orig_f,  deterministic)
      eps_z = jax.random.normal(self.make_rng('sample'), shape=mu_z.shape)
      embedding = mu_z + jnp.sqrt(var_z) * eps_z
      kl_z = 0.5 * jnp.sum(
        mu_z ** 2 + var_z - jnp.log(var_z) - 1.,
        axis=1)
    return embedding, kl_z

  def _get_score_model_gt(self, g_t):
    assert g_t.ndim == 4
    if self.config.unet_type == 'vdm':
      return jnp.mean(g_t, axis=(1 ,2, 3)).reshape(-1)
    elif self.config.unet_type in {'ldm', 'imp_samp'}:
      return g_t

  def __call__(
      self, images, labels, conditioning, step, deterministic: bool = True):
    x = images.reshape(-1, 32, 32, 3)
    n_batch = x.shape[0]

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
      embedding, kl_z = self._get_embedding_and_kl_z(
        orig_f, step=step, deterministic=deterministic)
    else:
      embedding = jax.nn.one_hot(labels, 10)
      kl_z = 0.0
    g_0 = self._get_gamma(embedding, jnp.zeros_like(t)).reshape(* orig_f.shape)
    g_1 = self._get_gamma(embedding, jnp.ones_like(t)).reshape(* orig_f.shape)
    g_t = self._get_gamma(embedding, t).reshape(* orig_f.shape)

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
      z_t, self._get_score_model_gt(g_t),
      conditioning, deterministic, time=False)
    if T == 0:
      _, g_t_grad = jax.jvp(
        self._get_gamma,
        (embedding, t),
        (jnp.zeros_like(embedding), jnp.ones_like(t)))
      g_t_grad = g_t_grad.reshape(* orig_f.shape)
      assert g_t_grad.shape == orig_f.shape
      loss_diff = .5 * jnp.sum(
        g_t_grad * jnp.square(eps - eps_hat),
        axis=[1, 2, 3])
    else:
      # loss for finite depth T, i.e. discrete time
      s = t - (1./T)
      g_s = self._get_gamma(embedding, s).reshape(* orig_f.shape)
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
      ones = jnp.ones((batch_size, self.config.latent_k))
      zeros = jnp.zeros((batch_size, k - self.config.latent_k))
      return jnp.concatenate([ones, zeros], axis=1)
    elif self.config.latent_type == 'gaussian':
      return jnp.zeros((batch_size, k))

  def conditional_sample(self, i, T, z_t, embedding, conditioning, rng): 
    rng_body = jax.random.fold_in(rng, i)
    eps = jax.random.normal(rng_body, z_t.shape)

    t = (T - i) / T
    s = (T - i - 1) / T
    t = t * jnp.ones((z_t.shape[0],), z_t.dtype)
    s = s * jnp.ones((z_t.shape[0],), z_t.dtype)

    g_t = self._get_gamma(embedding, t).reshape(* z_t.shape)
    g_s = self._get_gamma(embedding, s).reshape(* z_t.shape)

    if self.config.z_conditioning:
      conditioning = embedding
    else:
      conditioning = conditioning[:, None]

    eps_hat = self.score_model(
        z_t,
        self._get_score_model_gt(g_t),
        conditioning,
        deterministic=True)
  
    a = nn.sigmoid(-g_s)
    b = nn.sigmoid(-g_t)
    c = - jnp.expm1(g_s - g_t)
    sigma_t = jnp.sqrt(nn.sigmoid(g_t))
    z_s_mean = jnp.sqrt(a / b) * (z_t - sigma_t * c * eps_hat) 

    return z_s_mean + jnp.sqrt((1. - a) * c) * eps

  def sample(self, i, T, z_t, conditioning, rng): 
    rng_body = jax.random.fold_in(rng, i)
    eps = jax.random.normal(rng_body, z_t.shape)

    t = (T - i) / T
    s = (T - i - 1) / T
    t = t * jnp.ones((z_t.shape[0],), z_t.dtype)
    s = s * jnp.ones((z_t.shape[0],), z_t.dtype)
    
    embedding = self._get_deterministic_embedding(z_t.shape[0])

    g_t = self._get_gamma(embedding, t).reshape(* z_t.shape)
    g_s = self._get_gamma(embedding, s).reshape(* z_t.shape)

    if self.config.z_conditioning:
      conditioning = embedding
    else:
      conditioning = conditioning[:, None]

    eps_hat = self.score_model(
        z=z_t,
        g_t=self._get_score_model_gt(g_t),
        conditioning=conditioning,
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
      jnp.zeros((z_0.shape[0],))).reshape(* z_0.shape)
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

  def reverse_ode(self, xt, embeddings, t, high_precision=False):
    """Create the reverse-time ODE."""
    g_t = self._get_gamma(embeddings, t).reshape(* xt.shape)
    _, g_t_grad = jax.jvp(
      self._get_gamma,
      (embeddings, t),
      (jnp.zeros_like(embeddings), jnp.ones_like(t)))
    eps_hat = self.score_model(
        xt,
        self._get_score_model_gt(g_t),
        embeddings,
        deterministic=True)
    g_t_grad = g_t_grad.reshape(* xt.shape)
    if high_precision:
      sigma = jnp.where(nn.sigmoid(g_t) <= 1e-3,
                        jnp.exp(g_t / 2),
                        jnp.sqrt(nn.sigmoid(g_t)))
    else:
      sigma = jnp.sqrt(nn.sigmoid(g_t))
    return 0.5 * (- sigma * xt + eps_hat) * sigma * g_t_grad


class NoiseSchedule_polynomial_fixedend(nn.Module):
  config: model_vdm.VDMConfig
  n_features: int = 32 * 32 * 3
  nonlinear: bool = True
  n_sampling_timesteps = 1000

  def setup(self):
    n_out = 32 * 32 * 3
    self.min_gamma = self.config.gamma_min
    self.max_minus_min_gamma = self.config.gamma_max - self.config.gamma_min
    self.grad_min_epsilon = 0.

    self.l1 = nn.Dense(features=self.n_features, name='dense_1')
    self.l2 = nn.Dense(features=self.n_features, name='dense_2')
    self.l3_a = nn.Dense(
      features=n_out,
      name='dense_out_a',
      kernel_init=model_vdm.constant_init(0),
      bias_init=model_vdm.constant_init(0),
      )
    self.l3_b = nn.Dense(
      features=n_out,
      name='dense_out_b',
      # kernel_init=model_vdm.constant_init(0),
      # bias_init=model_vdm.constant_init(0),
      )
    self.l3_c = nn.Dense(
      features=n_out,
      name='dense_out_c',
      # kernel_init=model_vdm.constant_init(0),
      # bias_init=model_vdm.constant_init(0),
      )

  def _eval_polynomial(self, a, b, c, t):
    # derivative = (at^2 + bt + c)^2
    polynomial = (
      (a ** 2) * (t ** 5) / 5.0
      + (b ** 2 + 2 * a * c) * (t ** 3) / 3.0
      + a * b * (t ** 4) / 2.0
      + b * c * (t ** 2)
      + (c ** 2 + self.grad_min_epsilon) * t)
    
    scale = ((a ** 2) / 5.0
             + (b ** 2 + 2 * a * c) / 3.0
             + a * b / 2.0
             + b * c
             + (c ** 2 + self.grad_min_epsilon))

    return self.min_gamma + self.max_minus_min_gamma * polynomial / scale

  def _compute_coefficients(self, embedding):
    _h = nn.swish(self.l1(embedding))
    _h = nn.swish(self.l2(_h))
    a = self.l3_a(_h)
    # a = 1.0
    b = self.l3_b(_h)
    c = 1e-3 + nn.softplus(self.l3_c(_h))
    return a, b, c

  def _grad_t(self, a, b, c, t):
    # derivative = (at^2 + bt + c)^2
    polynomial = (
      (a ** 2) * (t ** 4)
      + (b ** 2 + 2 * a * c) * (t ** 2)
      + a * b * (t ** 3) * 2.0
      + b * c * t * 2
      + (c ** 2))
    
    scale = ((a ** 2) / 5.0
             + (b ** 2 + 2 * a * c) / 3.0
             + a * b / 2.0
             + b * c
             + (c ** 2))

    return self.max_minus_min_gamma * polynomial / scale

  def _discrete_gradient_all_points(self, embedding):
    a, b, c = self._compute_coefficients(embedding)
    # all_t = jnp.linspace(0, 1, num=self.n_sampling_timesteps + 1)[None, None, :]
    # gamma = self._eval_polynomial(
    #   a[:, :, None], b[:, :, None], c[:, :, None], all_t)
    # assert len(gamma.shape) == 3, f"{gamma.shape}"
    # return (gamma[:, :, 1:] - gamma[:, :, :-1]) * self.n_sampling_timesteps
    t = jnp.linspace(0, 1, num=self.n_sampling_timesteps)[None, None, :]
    return self._grad_t(a[:, :, None], b[:, :, None], c[:, :, None], t)

  def _interpolate(self, indices, cumulative_curve_length, targets):
    # https://adrian.pw/blog/flexible-density-model-jax/
    # https://pypi.org/project/sympy2jax/
    indices = indices[:, None]
    vals = jnp.take_along_axis(cumulative_curve_length, indices, axis=1)
    vals_mid = cumulative_curve_length[:, -1:] * targets[:, None]
    delta = vals - vals_mid
    indices_prime = (indices - 1) * (delta > 0) + (
      indices + 1) * (delta < 0) + indices * (delta == 0)
    vals_prime = jnp.take_along_axis(cumulative_curve_length, indices_prime, axis=1)
    delta_indices = (vals_mid - vals) * (indices - indices_prime) / (vals - vals_prime) 
    return jnp.squeeze(indices + delta_indices, axis=1)

  def inverse_sampling(self, embedding, targets):
    assert len(embedding.shape) == 2, f"{embedding.shape}"
    assert len(targets.shape) == 1, f"{targets.shape}"
    dgamma_dt = self._discrete_gradient_all_points(embedding)
    dl_dt = jnp.linalg.norm(dgamma_dt, ord=2, axis=1)
    
    assert len(dl_dt.shape) == 2, f"{dl_dt.shape}"
    dl_dt = 0.5 * (dl_dt[:, :-1] + dl_dt[:, 1:])
    cumulative_curve_length = jnp.cumsum(dl_dt, axis=1) / (self.n_sampling_timesteps - 1)
    cumulative_curve_length = jnp.pad(cumulative_curve_length, ((0, 0), (1, 0)))
    assert cumulative_curve_length.shape[1] == self.n_sampling_timesteps
    indices = jnp.argmin(jnp.square(
      cumulative_curve_length - cumulative_curve_length[:, -1:] * targets[:, None]), axis=1)
    assert indices.shape == targets.shape, f"indices: {indices.shape} targets: {targets.shape}"
    
    # linear interpolation.
    # indices = self._interpolate(indices, cumulative_curve_length, targets)

    new_t = indices.astype(float) / (self.n_sampling_timesteps - 1)
    integral_scaling_factor = (cumulative_curve_length[:, -1])
    return new_t, integral_scaling_factor

  def __call__(self, embedding, t, det_min_max=False):
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

    if jnp.isscalar(t) or len(t.shape) == 0:
      t = t * jnp.ones((embedding.shape[0], 1))
    else:
      t = jnp.reshape(t, (-1, 1))
    assert t.shape[0] == embedding.shape[0]
    assert len(embedding.shape) == 2, f"{embedding.shape}"

    a, b, c = self._compute_coefficients(embedding)
    return self._eval_polynomial(a, b, c, t)


class NoiseSchedule_NNet(nn.Module):
  config: model_vdm.VDMConfig
  n_features: int = 32 * 32 * 3

  def setup(self):
    config = self.config

    n_out_linear = 1
    n_out_nonlinear = 32 * 32 * 3
    kernel_init = nn.initializers.normal()

    init_bias = self.config.gamma_min
    init_scale = self.config.gamma_max - init_bias

    self.l1 = model_vdm.DenseMonotone(
      n_out_linear,
      kernel_init=model_vdm.constant_init(init_scale),
      bias_init=model_vdm.constant_init(init_bias))

    self.l2 = model_vdm.DenseMonotone(
      self.n_features, kernel_init=kernel_init)
    self.l_int = model_vdm.DenseMonotone(
      self.n_features, kernel_init=kernel_init)
    self.l3 = model_vdm.DenseMonotone(
      n_out_nonlinear, kernel_init=kernel_init, use_bias=False)

  @nn.compact
  def __call__(self, image_embedding, t, det_min_max=False):
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

    if jnp.isscalar(t) or len(t.shape) == 0:
      t = t * jnp.ones((1, 1))
    else:
      t = jnp.reshape(t, (-1, 1))
    assert t.shape[0] == image_embedding.shape[0]
    assert len(image_embedding.shape) == 2, f"{image_embedding.shape}"
    linear = self.l1(t)
    t = jnp.concatenate((image_embedding, t), axis=1)

    _h = 2. * (t - .5)  # scale input to [-1, +1]
    _h = self.l2(_h)
    _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
    _h = self.l_int(_h)
    _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
    _h = self.l3(_h) / self.n_features

    return linear + _h


class NoiseSchedule_FixedLinear(nn.Module):
  config: model_vdm.VDMConfig

  @nn.compact
  def __call__(self, image_embedding, t):
    config = self.config
    return config.gamma_min + (
        config.gamma_max - config.gamma_min) * t[:, None, None, None] * jnp.ones(
            (image_embedding.shape[0], 32, 32, 3))


GAMMA_NETWORKS = {
  'linear': NoiseSchedule_FixedLinear,
  'learnable_nnet': NoiseSchedule_NNet,
  'poly_fixedend': NoiseSchedule_polynomial_fixedend,
  }