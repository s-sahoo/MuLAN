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
# sbatch --gres=gpu:4 --exclude=ju-compute-01 run.sh -m ldm.main --config=ldm/configs/cifar10-small-discrete.py --workdir blur-noise

from flax import linen as nn
import jax
from jax import numpy as jnp
from ldm import model_vdm

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
    if self.config.reparam_type == 'mu_sigma':
      self.score_model = ScoreUNetMuSigma(self.config)
    else:
      self.score_model = model_vdm.ScoreUNet(self.config)
    self.encoder_model = ENCODER_MODELS[self.config.encoder](config=self.config)
    self.condition = self.config.condition
    if self.config.gamma_type == 'learnable_nnet':
      self.gamma_input_conditioned = NoiseSchedule_NNet(self.config)
    elif self.config.gamma_type == 'learnable_nnet_linear':
      self.gamma_input_conditioned = NoiseSchedule_NNet_Linear(self.config)
    self.epsilon = self.config.epsilon

  def apply_gamma(self, t):
    if jnp.isscalar(t) or len(t.shape) == 0:
      batch_size = 1
    else:
      batch_size = t.shape[0]
    return self._get_gamma_q(
      jnp.zeros((batch_size, 10)),
      jnp.zeros((batch_size, 32, 32, 3)),
      t,
      deterministic=True)[1]

  def _get_gamma_q(self, labels, x, t, deterministic):
    if self.condition == 'input':
      embedding = nn.sigmoid(self.encoder_model(x, deterministic))
    elif self.condition == 'label':
      embedding = jax.nn.one_hot(labels, 10)
    else:
      assert self.condition == 'ignore'
      embedding = jnp.ones((x.shape[0], 10), dtype=x.dtype)
    linear, nonlinear = self.gamma_input_conditioned(embedding, t)
    return jnp.squeeze(linear), jnp.squeeze(nonlinear)[:, None, None, None]

  def _get_gamma_p_theta(self, x, t, deterministic):
    embedding = self.encoder_model(x,  deterministic)
    if self.condition == 'input':
      embedding = nn.sigmoid(embedding)
    elif self.condition == 'label':
      embedding = jax.nn.softmax(embedding)
    else:
      assert self.condition == 'ignore'
      embedding = jnp.ones((x.shape[0], 10), dtype=x.dtype)
    linear, nonlinear = self.gamma_input_conditioned(embedding, t)
    return jnp.squeeze(linear), jnp.squeeze(nonlinear)[:, None, None, None]

  def _compute_mu_q_var_q(self, z_t, orig_f, g_s, g_t, epsilon):
    var_s = nn.sigmoid(g_s)
    var_t = nn.sigmoid(g_t)
    alpha_s = jnp.sqrt(nn.sigmoid(-g_s))
    alpha_t = jnp.sqrt(nn.sigmoid(-g_t))
    alpha_ts = alpha_t / alpha_s
    one_minus_snr_ts = - jnp.expm1(g_s - g_t)
    mu_q = (
      alpha_ts * var_s * z_t / var_t
      + alpha_s * one_minus_snr_ts * orig_f)
    var_q = epsilon + var_s * one_minus_snr_ts
    return mu_q, var_q

  def _compute_mu_theta_var_theta(
      self, z_t, approx_x, s, t, epsilon, deterministic):
    _, g_s_hat = self._get_gamma_p_theta(approx_x, s,  deterministic)
    var_s_hat = nn.sigmoid(g_s_hat)
    _, g_t_hat = self._get_gamma_p_theta(approx_x, t,  deterministic)
    var_t_hat = nn.sigmoid(g_t_hat)
    alpha_s_hat = jnp.sqrt(nn.sigmoid(-g_s_hat))
    alpha_t_hat = jnp.sqrt(nn.sigmoid(-g_t_hat))
    alpha_ts_hat = alpha_t_hat / alpha_s_hat
    one_minus_snr_ts_hat = - jnp.expm1(g_s_hat - g_t_hat)
    mu_theta = (
      alpha_ts_hat * var_s_hat * z_t / var_t_hat
      + alpha_s_hat * one_minus_snr_ts_hat * approx_x)
    var_theta = epsilon + var_s_hat * one_minus_snr_ts_hat
    assert g_s_hat.ndim == g_t_hat.ndim == var_theta.ndim == mu_theta.ndim == 4
    return mu_theta, var_theta

  def _get_model_predictions(self, z_t, g_t, t, conditioning, deterministic):
    if self.config.model_time:
      model_time = t
    else:
      model_time = g_t
    if self.config.reparam_type == 'mu_sigma':
      return self.score_model(
        z_t, model_time, conditioning, deterministic,
        time=self.config.model_time)
    model_output = self.score_model(
      z_t, model_time, conditioning, deterministic,
      time=self.config.model_time)
    if self.config.reparam_type  == 'input':
      _, g_t_hat = self._get_gamma_p_theta(
        model_output, t, deterministic)
      var_t_hat = nn.sigmoid(g_t_hat)
      # eps_hat
      model_output = (
        z_t - jnp.sqrt(1. - var_t_hat) * model_output) / jnp.sqrt(var_t_hat)
    return model_output, None

  def __call__(self, images, labels, conditioning, step, deterministic: bool = True):
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
    _, g_0 = self._get_gamma_q(labels, orig_f, jnp.zeros_like(t), deterministic)
    _, g_1 = self._get_gamma_q(labels, orig_f, jnp.ones_like(t), deterministic)
    linear_gamma, g_t = self._get_gamma_q(labels, orig_f, t, deterministic)
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

    # sample z_t
    eps = jax.random.normal(self.make_rng('sample'), shape=orig_f.shape)
    z_t = jnp.sqrt(1. - var_t) * orig_f + jnp.sqrt(var_t) * eps
    
    mu_theta, var_theta = self._get_model_predictions(
      z_t=z_t, g_t=linear_gamma, t=t, conditioning=conditioning,
      deterministic=deterministic)
    # jax.debug.print('var_theta {x}', x=var_theta)
    if T == 0:
      assert False
    else:
      # loss for finite depth T, i.e. discrete time
      s = t - (1./T)
      _, g_s = self._get_gamma_q(labels, orig_f, s, deterministic)

      if self.config.reparam_type in {'true', 'mu_sigma'}:
        epsilon = self.epsilon  # 1e-12, 1e-8
        if deterministic: # eval
          epsilon = 0.0
        mu_q, var_q = self._compute_mu_q_var_q(
          z_t=z_t, orig_f=orig_f, g_s=g_s, g_t=g_t, epsilon=epsilon)
        if self.config.reparam_type == 'true':
          mu_theta, var_theta = self._compute_mu_theta_var_theta(
            z_t=z_t, approx_x=mu_theta, s=s, t=t, epsilon=epsilon,
            deterministic=deterministic)
        assert mu_q.ndim == mu_theta.ndim == var_q.ndim == var_theta.ndim == 4
        loss_diff = 0.5 * T * jnp.sum(
          (jnp.square(mu_q - mu_theta) / var_theta
           + var_q / var_theta
           - jnp.log(var_q) + jnp.log(var_theta)
           - 1.0), 
          axis=[1, 2, 3])
      else:
        snr = jnp.expm1(g_t - g_s)
        loss_diff = 0.5 * T * jnp.sum(
          jnp.square(eps - mu_theta) * snr,
          axis=[1, 2, 3])
    return model_vdm.VDMOutput(
        loss_recon=loss_recon,
        loss_klz=loss_klz,
        loss_diff=loss_diff,
        var_0=jnp.mean(var_0),
        var_1=jnp.mean(var_1),
    )

  def sample(self, i, T, z_t, conditioning, rng): 
    rng_body = jax.random.fold_in(rng, i)
    eps = jax.random.normal(rng_body, z_t.shape)

    t = (T - i) / T
    s = (T - i - 1) / T
    if self.config.model_time:
      model_output = self.score_model(
          z_t,
          t * jnp.ones((z_t.shape[0],), t.dtype),
          conditioning,
          deterministic=True,
          time=True)
      if self.config.reparam_type == 'mu_sigma':
        return model_output[0] + model_output[1] * eps
    else:
      linear_gamma, _ = self._get_gamma_q(
        jnp.zeros((z_t.shape[0],)), jnp.zeros_like(z_t), t,
        deterministic=True)
      model_output = self.score_model(
          z_t,
          linear_gamma * jnp.ones((z_t.shape[0],), z_t.dtype),
          conditioning,
          deterministic=True)
      if self.config.reparam_type == 'mu_sigma':
        return model_output[0] + model_output[1] * eps
    _, g_s = self._get_gamma_p_theta(model_output, s, deterministic=True)
    _, g_t = self._get_gamma_p_theta(model_output, t, deterministic=True)
    if self.config.reparam_type == 'noise':
      eps_hat = model_output
    elif self.config.reparam_type in {'input', 'true'}:
      var_t = nn.sigmoid(g_t)
      eps_hat = (
        z_t - jnp.sqrt(1. - var_t) * model_output) / jnp.sqrt(var_t)
  
    a = nn.sigmoid(-g_s)
    b = nn.sigmoid(-g_t)
    c = - jnp.expm1(g_s - g_t)
    sigma_t = jnp.sqrt(nn.sigmoid(g_t))
    z_s_mean = jnp.sqrt(a / b) * (z_t - sigma_t * c * eps_hat) 

    return z_s_mean + jnp.sqrt((1. - a) * c) * eps

  def generate_x(self, z_0):
    _, g_0 = self._get_gamma_q(
      jnp.zeros((z_0.shape[0],)),
      z_0,
      jnp.zeros((z_0.shape[0],)),
      deterministic=True)
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


class DenseMonotoneNew(nn.Dense):
  '''Strictly increasing Dense layer.'''

  @nn.compact
  def __call__(self, inputs):
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = self.param('kernel',
                        self.kernel_init,
                        (inputs.shape[-1], self.features))
    kernel = jax.nn.softplus(jnp.asarray(kernel, self.dtype))
    y = jax.lax.dot_general(inputs, kernel,
                            (((inputs.ndim - 1,), (0,)), ((), ())),
                            precision=self.precision)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y

class NoiseSchedule_NNet_Linear(nn.Module):
  config: model_vdm.VDMConfig
  n_features: int = 1024

  def setup(self):
    config = self.config

    n_out = 1
    kernel_init = nn.initializers.normal()

    self.l2 = model_vdm.DenseMonotone(self.n_features, kernel_init=kernel_init)
    self.l3 = model_vdm.DenseMonotone(n_out, kernel_init=kernel_init, use_bias=False)

  @nn.compact
  def __call__(self, image_embedding, t, det_min_max=False):
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

    if jnp.isscalar(t) or len(t.shape) == 0:
      t = t * jnp.ones((image_embedding.shape[0], 1))
    else:
      t = jnp.reshape(t, (-1, 1))
    assert t.shape[0] == image_embedding.shape[0]
    assert len(image_embedding.shape) == 2
    linear = self.config.gamma_min + t * (self.config.gamma_max - self.config.gamma_min)

    t_t = jnp.concatenate((image_embedding, t), axis=1)
    t_0 = jnp.concatenate((image_embedding, jnp.zeros_like(t)), axis=1)
    t_1 = jnp.concatenate((image_embedding, jnp.ones_like(t)), axis=1)
    
    _h_t = 2. * (t_t - .5)  # scale input to [-1, +1]
    _h_t = self.l2(_h_t)
    _h_t = 2 * (nn.sigmoid(_h_t) - .5)  # more stable than jnp.tanh(h)
    _h_t = self.l3(_h_t) / self.n_features

    _h_0 = 2. * (t_0 - .5)  # scale input to [-1, +1]
    _h_0 = self.l2(_h_0)
    _h_0 = 2 * (nn.sigmoid(_h_0) - .5)  # more stable than jnp.tanh(h)
    _h_0 = self.l3(_h_0) / self.n_features

    _h_1 = 2. * (t_1 - .5)  # scale input to [-1, +1]
    _h_1 = self.l2(_h_1)
    _h_1 = 2 * (nn.sigmoid(_h_1) - .5)  # more stable than jnp.tanh(h)
    _h_1 = self.l3(_h_1) / self.n_features

    _h_scaled = (_h_t - _h_0) / (_h_1 - _h_0)
    _h = self.config.gamma_min + _h_scaled * (
      self.config.gamma_max - self.config.gamma_min)
    return linear, (linear + _h) / 2

class NoiseSchedule_Film(nn.Module):
  config: model_vdm.VDMConfig
  n_features: int = 1024

  def setup(self):
    config = self.config

    n_out = 1
    kernel_init = nn.initializers.normal()

    init_bias = self.config.gamma_min
    init_scale = self.config.gamma_max - init_bias
  
    self.l1 = model_vdm.DenseMonotone(
      n_out,
      kernel_init=model_vdm.constant_init(init_scale),
      bias_init=model_vdm.constant_init(init_bias))
    self.l2 = model_vdm.DenseMonotone(self.n_features, kernel_init=kernel_init)
    self.l_int = model_vdm.DenseMonotone(self.n_features, kernel_init=kernel_init)
    self.l3 = model_vdm.DenseMonotone(n_out, kernel_init=kernel_init, use_bias=False)
    
    self.film_dense1 = nn.Dense(features=self.n_features, name='film_1')
    self.film_dense2 = nn.Dense(features=self.n_features, name='film_2')
    # gamma network has 2 layers
    self.film_dense3_scale = nn.Dense(
      features=self.n_features, name='film_3_scale')
    self.film_dense3_bias = nn.Dense(
      features=self.n_features, name='film_3_bias')


  @nn.compact
  def __call__(self, image_embedding, t, det_min_max=False):
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

    if jnp.isscalar(t) or len(t.shape) == 0:
      t = t * jnp.ones((image_embedding.shape[0], 1))
    else:
      t = jnp.reshape(t, (-1, 1))
    assert t.shape[0] == image_embedding.shape[0]
    assert len(image_embedding.shape) == 2
    linear = self.l1(t)

    # t = jnp.concatenate((image_embedding, t), axis=1)
    film_activations = nn.swish(self.film_dense1(image_embedding))
    film_activations = nn.swish(self.film_dense2(film_activations))
    film_scale = nn.softplus(self.film_dense3_scale(film_activations))
    film_bias = self.film_dense3_bias(film_activations)
    assert film_scale.shape == (image_embedding.shape[0], self.n_features)
    assert film_bias.shape == (image_embedding.shape[0], self.n_features)

    # t -> l2 -> film_layer -> l3
    _h = 2. * (t - .5)  # scale input to [-1, +1]
    _h = self.l2(_h)
    _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
    _h = _h  * film_scale + film_bias
    _h = self.l_int(_h)
    _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
    _h = self.l3(_h) / self.n_features

    return linear, linear + _h 

class NoiseSchedule_NNet(nn.Module):
  config: model_vdm.VDMConfig
  n_features: int = 1024

  def setup(self):
    config = self.config

    n_out = 1
    kernel_init = nn.initializers.normal()

    init_bias = self.config.gamma_min
    init_scale = self.config.gamma_max - init_bias

    self.l1 = model_vdm.DenseMonotone(
      n_out,
      kernel_init=model_vdm.constant_init(init_scale),
      bias_init=model_vdm.constant_init(init_bias))
    self.l2 = model_vdm.DenseMonotone(self.n_features, kernel_init=kernel_init)
    self.l3 = model_vdm.DenseMonotone(n_out, kernel_init=kernel_init, use_bias=False)

  @nn.compact
  def __call__(self, image_embedding, t, det_min_max=False):
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

    if jnp.isscalar(t) or len(t.shape) == 0:
      t = t * jnp.ones((image_embedding.shape[0], 1))
    else:
      t = jnp.reshape(t, (-1, 1))
    assert t.shape[0] == image_embedding.shape[0]
    assert len(image_embedding.shape) == 2
    linear = self.l1(t)

    t = jnp.concatenate((image_embedding, t), axis=1)
    
    _h = 2. * (t - .5)  # scale input to [-1, +1]
    _h = self.l2(_h)
    _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
    _h = self.l3(_h) / self.n_features

    return linear, linear + _h 


class ScoreUNetMuSigma(nn.Module):
  config: model_vdm.VDMConfig
  n_layer: int = -1

  @nn.compact
  def __call__(self, z, g_t, conditioning, deterministic=True, time=False):
    config = self.config
    if self.n_layer == -1:
      n_layers = self.config.sm_n_layer
    else:
      n_layers = self.n_layer
    # Compute conditioning vector based on 'g_t' and 'conditioning'
    n_embd = self.config.sm_n_embd
    if time:
      t = g_t
    else:
      lb = config.gamma_min
      ub = config.gamma_max
      t = (g_t - lb) / (ub - lb)  # ---> [0,1]

    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
    if jnp.isscalar(t):
      t = jnp.ones((z.shape[0],), z.dtype) * t
    elif len(t.shape) == 0:
      t = jnp.tile(t[None], z.shape[0])

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
    for i_block in range(n_layers):
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

    # Upsampling
    for i_block in range(n_layers + 1):
      b = model_vdm.ResnetBlock(config, out_ch=n_embd, name=f'up.block_{i_block}')
      h = b(jnp.concatenate([h, hs.pop()], axis=-1), cond, deterministic)[0]
      if config.with_attention:
        h = model_vdm.AttnBlock(num_heads=1, name=f'up.attn_{i_block}')(h)

    assert not hs

    # Predict noise
    normalize = nn.normalization.GroupNorm()
    h = nn.swish(normalize(h))
    mu_pred = nn.Conv(
        features=z.shape[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv_out')(h)
    # Base measure
    mu_pred += z

    sigma_pred = nn.Dense(1, name='dense_out_sigma')(h)

    return mu_pred, jnp.exp(sigma_pred)

class NoiseSchedule_polynomial(nn.Module):
  config: model_vdm.VDMConfig
  n_features: int = 32 * 32 * 3
  nonlinear: bool = True

  def setup(self):
    n_out = 1
    self.min_gamma = self.param(
      'min_gamma', model_vdm.constant_init(self.config.gamma_min),
      (n_out,))
    self.max_minus_min_gamma = self.param(
      'max_minus_min_gamma',
      model_vdm.constant_init(self.config.gamma_max - self.config.gamma_min),
      (n_out,))

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


  @nn.compact
  def __call__(self, image_embedding, t, det_min_max=False):
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

    if jnp.isscalar(t) or len(t.shape) == 0:
      t = t * jnp.ones((image_embedding.shape[0], 1))
    else:
      t = jnp.reshape(t, (-1, 1))
    assert t.shape[0] == image_embedding.shape[0]
    assert len(image_embedding.shape) == 2, f"{image_embedding.shape}"
    _h = nn.swish(self.l1(image_embedding))
    _h = nn.swish(self.l2(_h))
    a = self.l3_a(_h)
    # a = 1.0
    b = self.l3_b(_h)
    c = 1e-3 + nn.softplus(self.l3_c(_h))

    # derivative = (at^2 + bt + c)^2
    integral = (
      (a ** 2) * (t ** 5) / 5.0
      + (b ** 2 + 2 * a * c) * (t ** 3) / 3.0
      + a * b * (t ** 4) / 2.0
      + b * c * (t ** 2)
      + (c ** 2) * t)
    
    scaled_integral = integral / (
      (a ** 2) / 5.0
      + (b ** 2 + 2 * a * c) / 3.0
      + a * b / 2.0
      + b * c
      + c ** 2)

    return None, self.min_gamma + self.max_minus_min_gamma * scaled_integral