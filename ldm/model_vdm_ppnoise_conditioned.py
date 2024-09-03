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
import numpy as np

from ldm import model_vdm
from ldm import model_vdm_conditioned


class VDM(model_vdm.VDM):
  config: model_vdm.VDMConfig

  def setup(self):
    self.encdec = model_vdm.EncDec(self.config)
    self.score_model = model_vdm.ScoreUNet(self.config)
    self.encoder_model = model_vdm_conditioned.ENCODER_MODELS[self.config.encoder](self.config)
    self.condition = self.config.condition
    self.gamma = GAMMA_NETWORKS[self.config.gamma_type](self.config)
    self.epsilon = self.config.epsilon

  def apply_gamma(self, t, x_zero=None):
    # TODO: this function should accept images and labels.
    if jnp.isscalar(t) or len(t.shape) == 0:
      batch_size = 1
    else:
      batch_size = t.shape[0]
    if x_zero is None:
      x_zero = jnp.zeros((batch_size, 32, 32, 3))
    return self._get_gamma_q(
      jnp.zeros((batch_size, 10)),
      x_zero,
      t)

  def _get_gamma_q(self, labels, x, t, deterministic):
    if self.condition == 'input':
      embedding = nn.sigmoid(self.encoder_model(x, deterministic))
    elif self.condition == 'label':
      embedding = jax.nn.one_hot(labels, 10)
    else:
      assert self.condition == 'ignore'
      embedding = jnp.ones((x.shape[0], 10), dtype=x.dtype)
    return self.gamma(embedding, t)

  def _get_gamma_p_theta(self, x, t, deterministic):
    embedding = self.encoder_model(x, deterministic)
    if self.condition == 'input':
      embedding = nn.sigmoid(embedding)
    elif self.condition == 'label':
      embedding = jax.nn.softmax(embedding)
    else:
      assert self.condition == 'ignore'
      embedding = jnp.ones((x.shape[0], 10), dtype=x.dtype)
    return self.gamma(embedding, t)

  def sample_timesteps(self, n_batch):
    rng1 = self.make_rng('sample')
    if self.config.antithetic_time_sampling:
      t0 = jax.random.uniform(rng1)
      t = jnp.mod(t0 + jnp.arange(0., 1., step=1. / n_batch), 1.)
    else:
      t = jax.random.uniform(rng1, shape=(n_batch,))
    return t

  def reconstruction_loss(self, orig_f, x, g_0):
    # 1. Reconstruction loss
    f_0 = orig_f
    eps_0 = jax.random.normal(
      self.make_rng('sample'), shape=f_0.shape)
    # z_0 = jnp.sqrt(1. - var_0) * f_0 + jnp.sqrt(var_0) * eps_0
    z_0_rescaled = f_0 + jnp.exp(0.5 * g_0) * eps_0  # = z_0 / sqrt(1 - var)
    return - self.encdec.logprob(x, z_0_rescaled, g_0)

  def kl_loss(self, orig_f, var_1):
    # KL z1 with N(0,1) prior
    f_1 = orig_f
    mu_sqr = (1. - var_1) * jnp.square(f_1)
    return 0.5 * jnp.sum(
      mu_sqr + var_1 - jnp.log(var_1) - 1.,
      axis=(1, 2, 3))

  def discrete_loss_diff(
      self, eps, eps_hat, g_t, t, T, gamma_function):
      s = t - (1./T)
      g_t = g_t.reshape(* eps.shape)
      g_s = gamma_function(s).reshape(* eps.shape)
      del_snr = jnp.expm1(g_t - g_s)
      del_eps_square = jnp.square(eps - eps_hat)
      return 0.5 * T * jnp.sum(
        del_eps_square * del_snr, axis=[1, 2, 3])

  def __call__(self, images, labels, conditioning, step, deterministic: bool=True):
    batch_size = images.shape[0]
    t = self.sample_timesteps(batch_size)
    x = images

    # 3. DIFFUSION LOSS
    # sample time steps

    # discretize time steps if we're working with discrete time
    T = self.config.sm_n_timesteps
    if T > 0:
      t = jnp.ceil(t * T) / T

    # Process input
    orig_f = self.encdec.encode(x)
    f = orig_f
    gamma_function = lambda time: self._get_gamma_q(
      labels, f, time, deterministic=deterministic)
  
    g_0 = gamma_function(jnp.zeros_like(t))
    g_1 = gamma_function(jnp.ones_like(t))
    var_0 = nn.sigmoid(g_0)
    var_1 = nn.sigmoid(g_1)

    g_0 = g_0.reshape(* f.shape)
    g_1 = g_1.reshape(* f.shape)
    var_0 = var_0.reshape(* f.shape)
    var_1 = var_1.reshape(* f.shape)
    # 1. Reconstruction loss
    loss_recon = self.reconstruction_loss(orig_f, x, g_0)
    # 2. Latent Loss
    loss_klz = self.kl_loss(orig_f, var_1)

    # sample z_t
    g_t = gamma_function(t).reshape(* f.shape)
    var_t = nn.sigmoid(g_t)
    eps = jax.random.normal(self.make_rng('sample'), shape=f.shape)
    z_t = jnp.sqrt(1. - var_t) * f + jnp.sqrt(var_t) * eps
    
    # 3. Intermediate Loss
    model_output = self.score_model(
      z_t, t, conditioning, deterministic, time=True)
    if self.config.reparam_type == 'noise':
      eps_hat = model_output
    elif self.config.reparam_type in {'input', 'true'}:
      g_t_hat = self._get_gamma_p_theta(
        model_output, t, deterministic=deterministic).reshape(* f.shape)
      var_t_hat = nn.sigmoid(g_t_hat)
      eps_hat = (
        z_t - jnp.sqrt(1. - var_t_hat) * model_output) / jnp.sqrt(var_t_hat)

    if T == 0:
      assert False
    else:
      # loss for finite depth T, i.e. discrete time
      if self.config.reparam_type == 'true':
        epsilon = self.epsilon
        if deterministic: # eval
          epsilon = 0.0
        s = t - (1./T)
        g_s = gamma_function(s).reshape(* f.shape)
        var_s = nn.sigmoid(g_s)
        alpha_s = jnp.sqrt(nn.sigmoid(-g_s))
        alpha_t = jnp.sqrt(nn.sigmoid(-g_t))
        alpha_ts = alpha_t / alpha_s
        one_minus_snr_st = - jnp.expm1(g_s - g_t)
        mu_q = alpha_ts * var_s * z_t / var_t + alpha_s * one_minus_snr_st * orig_f
        var_q = epsilon + var_s * one_minus_snr_st

        g_s_hat = self._get_gamma_p_theta(
          model_output, s, deterministic=deterministic).reshape(* f.shape)
        var_s_hat = nn.sigmoid(g_s_hat)
        alpha_s_hat = jnp.sqrt(nn.sigmoid(-g_s_hat))
        alpha_t_hat = jnp.sqrt(nn.sigmoid(-g_t_hat))
        alpha_ts_hat = alpha_t_hat / alpha_s_hat
        one_minus_snr_st_hat = - jnp.expm1(g_s_hat - g_t_hat)
        mu_theta = (
          alpha_ts_hat * var_s_hat * z_t / var_t_hat
          + alpha_s_hat * one_minus_snr_st_hat * model_output)
        var_theta = epsilon + var_s_hat * one_minus_snr_st_hat
        # var_theta = var_q
        assert mu_q.ndim == mu_theta.ndim == var_q.ndim == var_theta.ndim == 4
        assert g_s_hat.ndim == g_t_hat.ndim == var_q.ndim == var_theta.ndim == 4
        loss_diff = 0.5 * T * jnp.sum(
          (jnp.square(mu_q - mu_theta) / var_theta
           + var_q / var_theta
           - jnp.log(var_q) + jnp.log(var_theta)
           - 1.0), 
          axis=[1, 2, 3])
      else:
        assert False
        loss_diff = self.discrete_loss_diff(
          eps, eps_hat, g_t, t, T, gamma_function)

    # jax.debug.print('loss_recon {x}', x=loss_recon)
    return model_vdm.VDMOutput(
        loss_recon=loss_recon,
        loss_klz=loss_klz,
        loss_diff=loss_diff,
        var_0=jnp.mean(var_0),
        var_1=jnp.mean(var_1),
    )

  def sample(self, i, T, z_t, conditioning, rng):
    delta = 1e-8
    batch_size = z_t.shape[0]
    rng_body = jax.random.fold_in(rng, i)
    t = (T - i) / T
    s = (T - i - 1) / T

    model_output = self.score_model(
        z_t,
        t * jnp.ones((z_t.shape[0],), t.dtype),
        conditioning,
        deterministic=True,
        time=True)
    g_s = self._get_gamma_p_theta(
      model_output,
      s * jnp.ones((batch_size,)),
      deterministic=True).reshape(* z_t.shape)
    g_t = self._get_gamma_p_theta(
      model_output,
      t * jnp.ones((batch_size,)),
      deterministic=True).reshape(* z_t.shape)
    if self.config.reparam_type == 'noise':
      eps_hat = model_output
    elif self.config.reparam_type in {'input', 'true'}:
      var_t = nn.sigmoid(g_t)
      eps_hat = (
        z_t - jnp.sqrt(1. - var_t) * model_output) / jnp.sqrt(var_t)
  
    sigma_t = jnp.sqrt(nn.sigmoid(g_t))
    alpha_t = jnp.sqrt(nn.sigmoid(-g_t))

    sigma_s = jnp.sqrt(nn.sigmoid(g_s))
    alpha_s = jnp.sqrt(nn.sigmoid(-g_s))

    alpha_st = alpha_s / jnp.clip(alpha_t, a_min=delta)
    snr_ts = jnp.expm1(g_s - g_t) + 1
    sigma_denoise = jnp.sqrt(1 - snr_ts) * sigma_s

    # Compute terms.
    u_t = z_t
    coeff_term2 = alpha_st * (1 - snr_ts) * sigma_t
    mu_denoise = alpha_st * u_t - coeff_term2 * eps_hat
    eps = jax.random.normal(rng_body, z_t.shape)
  
    return mu_denoise + sigma_denoise * eps

  def generate_x(self, z_0):
    g_0 = self._get_gamma_p_theta(
      z_0, jnp.zeros((z_0.shape[0],)),
      deterministic=True).reshape(* z_0.shape)
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


######### Noise Schedule #########

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


class DenseMonotoneSoftplus(nn.Dense):
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


class NoiseSchedule_NNet_TimeMatrix(nn.Module):
  config: model_vdm.VDMConfig
  n_features: int = 1024
  nonlinear: bool = True

  def setup(self):
    if self.config.monotone_layer == 'dense_monotone':
      monotone_layer = model_vdm.DenseMonotone
    elif self.config.monotone_layer == 'dense_monotone_softplus':
      monotone_layer = DenseMonotoneSoftplus
    kernel_init = nn.initializers.normal()
    n_out = 1
    init_bias = self.config.gamma_min
    init_scale = self.config.gamma_max - init_bias

    self.l1 = monotone_layer(
      n_out,
      kernel_init=model_vdm.constant_init(init_scale),
      bias_init=model_vdm.constant_init(init_bias))
    if self.nonlinear:
      self.l2 = monotone_layer(
        self.n_features, kernel_init=kernel_init)
      self.l_int = monotone_layer(
        self.n_features, kernel_init=kernel_init)
      self.l3 = monotone_layer(
        n_out, kernel_init=kernel_init, use_bias=False)

    n_out = 32 * 32 * 3

    self.d1 = monotone_layer(
      n_out,
      kernel_init=model_vdm.constant_init(0),
      bias_init=model_vdm.constant_init(0))
    if self.nonlinear:
      self.d2 = monotone_layer(
        self.n_features, kernel_init=kernel_init)
      self.d_int = monotone_layer(
        self.n_features, kernel_init=kernel_init)
      self.d3 = monotone_layer(
        n_out, kernel_init=kernel_init, use_bias=False)

  @nn.compact
  def __call__(self, image_embedding, t, det_min_max=False):
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

    if jnp.isscalar(t) or len(t.shape) == 0:
      t = t * jnp.ones((image_embedding.shape[0], 1))
    else:
      t = jnp.reshape(t, (-1, 1))
    assert t.shape[0] == image_embedding.shape[0]
    assert len(image_embedding.shape) == 2, f"{image_embedding.shape}"
    h = self.l1(t)
    if self.nonlinear:
      _h = 2. * (t - .5)  # scale input to [-1, +1]
      _h = self.l2(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.l_int(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.l3(_h) / self.n_features
      h += _h
    
    orig = h
    t = jnp.concatenate((image_embedding, t), axis=1)

    h = self.d1(t)
    if self.nonlinear:
      _h = 2. * (t - .5)  # scale input to [-1, +1]
      _h = self.d2(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.d_int(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.d3(_h) / self.n_features
      h += _h
    return orig + h


class NoiseSchedule_NNet_TimeMatrix_Film(nn.Module):
  config: model_vdm.VDMConfig
  n_features: int = 1024
  nonlinear: bool = True

  def setup(self):
    if self.config.monotone_layer == 'dense_monotone':
      monotone_layer = model_vdm.DenseMonotone
    elif self.config.monotone_layer == 'dense_monotone_softplus':
      monotone_layer = DenseMonotoneSoftplus
    kernel_init = nn.initializers.normal()
    n_out = 1
    init_bias = self.config.gamma_min
    init_scale = self.config.gamma_max - init_bias

    self.l1 = monotone_layer(
      n_out,
      kernel_init=model_vdm.constant_init(init_scale),
      bias_init=model_vdm.constant_init(init_bias))
    if self.nonlinear:
      self.l2 = monotone_layer(
        self.n_features, kernel_init=kernel_init)
      self.l_int = monotone_layer(
        self.n_features, kernel_init=kernel_init)
      self.l3 = monotone_layer(
        n_out, kernel_init=kernel_init, use_bias=False)

    n_out = 32 * 32 * 3

    self.d1 = monotone_layer(
      n_out,
      kernel_init=model_vdm.constant_init(0),
      bias_init=model_vdm.constant_init(0))
    if self.nonlinear:
      self.d2 = monotone_layer(
        self.n_features, kernel_init=kernel_init)
      self.d_int = monotone_layer(
        self.n_features, kernel_init=kernel_init)
      self.d3 = monotone_layer(
        n_out, kernel_init=kernel_init, use_bias=False)

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
    assert len(image_embedding.shape) == 2, f"{image_embedding.shape}"
    film_activations = nn.swish(self.film_dense1(image_embedding))
    film_activations = nn.swish(self.film_dense2(film_activations))
    film_scale = nn.softplus(self.film_dense3_scale(film_activations))
    film_bias = self.film_dense3_bias(film_activations)
    assert film_scale.shape == (image_embedding.shape[0], self.n_features)
    assert film_bias.shape == (image_embedding.shape[0], self.n_features)

    # scalar: linear + non-linear
    h = self.l1(t)
    if self.nonlinear:
      _h = 2. * (t - .5)  # scale input to [-1, +1]
      _h = self.l2(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = _h * film_scale + film_bias
      _h = self.l_int(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.l3(_h) / self.n_features
      h += _h
    
    orig = h
    t = jnp.concatenate((image_embedding, t), axis=1)
    
    # pp: linear + non-linear
    h = self.d1(t)
    if self.nonlinear:
      _h = 2. * (t - .5)  # scale input to [-1, +1]
      _h = self.d2(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = _h  * film_scale + film_bias
      _h = self.d_int(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.d3(_h) / self.n_features
      h += _h
    return orig + h

class NoiseSchedule_NNet_TimeMatrix_FilmV2(nn.Module):
  config: model_vdm.VDMConfig
  n_features: int = 1024
  nonlinear: bool = True

  def setup(self):
    if self.config.monotone_layer == 'dense_monotone':
      monotone_layer = model_vdm.DenseMonotone
    elif self.config.monotone_layer == 'dense_monotone_softplus':
      monotone_layer = DenseMonotoneSoftplus
    kernel_init = nn.initializers.normal()
    n_out = 1
    init_bias = self.config.gamma_min
    init_scale = self.config.gamma_max - init_bias

    self.l1 = monotone_layer(
      n_out,
      kernel_init=model_vdm.constant_init(init_scale),
      bias_init=model_vdm.constant_init(init_bias))
    if self.nonlinear:
      self.l2 = monotone_layer(
        self.n_features, kernel_init=kernel_init)
      self.l_int = monotone_layer(
        self.n_features, kernel_init=kernel_init)
      self.l3 = monotone_layer(
        n_out, kernel_init=kernel_init, use_bias=False)

    n_out = 32 * 32 * 3

    self.d1 = monotone_layer(
      n_out,
      kernel_init=model_vdm.constant_init(0),
      bias_init=model_vdm.constant_init(0))
    if self.nonlinear:
      self.d2 = monotone_layer(
        self.n_features, kernel_init=kernel_init)
      self.d_int = monotone_layer(
        self.n_features, kernel_init=kernel_init)
      self.d3 = monotone_layer(
        n_out, kernel_init=kernel_init, use_bias=False)

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
    assert len(image_embedding.shape) == 2, f"{image_embedding.shape}"
    film_activations = nn.swish(self.film_dense1(image_embedding))
    film_activations = nn.swish(self.film_dense2(film_activations))
    film_scale = nn.softplus(self.film_dense3_scale(film_activations))
    film_bias = self.film_dense3_bias(film_activations)
    assert film_scale.shape == (image_embedding.shape[0], self.n_features)
    assert film_bias.shape == (image_embedding.shape[0], self.n_features)

    # scalar: linear + non-linear
    h = self.l1(t)
    if self.nonlinear:
      _h = 2. * (t - .5)  # scale input to [-1, +1]
      _h = self.l2(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = _h * film_scale + film_bias
      _h = self.l_int(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.l3(_h) / self.n_features
      h += _h
    
    orig = h
    
    # pp: linear + non-linear
    h = self.d1(t)
    if self.nonlinear:
      _h = 2. * (t - .5)  # scale input to [-1, +1]
      _h = self.d2(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = _h  * film_scale + film_bias
      _h = self.d_int(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.d3(_h) / self.n_features
      h += _h
    return orig + h


class NoiseSchedule_NNet_v2(nn.Module):
  config: model_vdm.VDMConfig
  n_features: int = 1024
  nonlinear: bool = True

  def setup(self):
    if self.config.monotone_layer == 'dense_monotone':
      monotone_layer = model_vdm.DenseMonotone
    elif self.config.monotone_layer == 'dense_monotone_softplus':
      monotone_layer = DenseMonotoneSoftplus
    kernel_init = nn.initializers.normal()
    n_out = 1
    init_bias = self.config.gamma_min
    init_scale = self.config.gamma_max - init_bias

    self.l1 = monotone_layer(
      n_out,
      kernel_init=model_vdm.constant_init(init_scale),
      bias_init=model_vdm.constant_init(init_bias))
    self.l2 = monotone_layer(
      self.n_features, kernel_init=kernel_init)
    self.l_int = monotone_layer(
      self.n_features, kernel_init=kernel_init)
    self.l3 = monotone_layer(
      n_out, kernel_init=kernel_init, use_bias=False)

    n_out = 32 * 32 * 3

    self.d1 = monotone_layer(
      n_out,
      kernel_init=model_vdm.constant_init(0),
      bias_init=model_vdm.constant_init(0))

    self.linearembedding = nn.Dense(
      features=self.n_features, name='linear_embedding_layer')
    self.d2 = monotone_layer(
      self.n_features, kernel_init=kernel_init)
    self.d_int = monotone_layer(
      self.n_features, kernel_init=kernel_init)
    self.d3 = monotone_layer(
      n_out, kernel_init=kernel_init, use_bias=False)

  @nn.compact
  def __call__(self, image_embedding, t, det_min_max=False):
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

    if jnp.isscalar(t) or len(t.shape) == 0:
      t = t * jnp.ones((image_embedding.shape[0], 1))
    else:
      t = jnp.reshape(t, (-1, 1))
    assert t.shape[0] == image_embedding.shape[0]
    assert len(image_embedding.shape) == 2, f"{image_embedding.shape}"
    h = self.l1(t)
    if self.nonlinear:
      _h = 2. * (t - .5)  # scale input to [-1, +1]
      _h = self.l2(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.l_int(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.l3(_h) / self.n_features
      h += _h
    
    orig = h

    h = self.d1(t)
    if self.nonlinear:
      _h = 2. * (t - .5)  # scale input to [-1, +1]
      _h = self.d2(_h) + self.linearembedding(image_embedding)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.d_int(_h)
      _h = 2 * (nn.sigmoid(_h) - .5)  # more stable than jnp.tanh(h)
      _h = self.d3(_h) / self.n_features
      h += _h
    return orig + h


class NoiseSchedule_Sigmoid(nn.Module):
  config: model_vdm.VDMConfig
  n_features: int = 32 * 32 * 3
  nonlinear: bool = True

  def setup(self):
    n_out = 32 * 32 * 3

    self.min_gamma = self.param(
      'min_gamma', model_vdm.constant_init(self.config.gamma_min),
      (n_out,))
    self.max_minus_min_gamma = self.param(
      'max_minus_min_gamma',
      model_vdm.constant_init(self.config.gamma_max - self.config.gamma_min),
      (n_out,))

    self.l1 = nn.Dense(features=self.n_features, name='dense_1')
    self.l2 = nn.Dense(features=self.n_features, name='dense_2')
    self.l3_w = nn.Dense(features=n_out, name='dense_out_w')
    self.l3_b = nn.Dense(features=n_out, name='dense_out_b')


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
    w = 0.01 + nn.softplus(self.l3_w(_h))
    b = jnp.clip(self.l3_b(_h), -3, 3)

    out_t = nn.sigmoid(w * t + b)
    out_0 = nn.sigmoid(b)
    out_1 = nn.sigmoid(w + b)

    g_t = self.min_gamma[None, :] + self.max_minus_min_gamma[None, :] * (
        out_t - out_0) / (out_1 - out_0)

    return g_t

class NoiseSchedule_Sigmoid2(nn.Module):
  config: model_vdm.VDMConfig
  n_features: int = 32 * 32 * 3
  nonlinear: bool = True

  def setup(self):
    n_out = 32 * 32 * 3
    self.linear = model_vdm.DenseMonotone(
      n_out,
      kernel_init=model_vdm.constant_init(
        self.config.gamma_max - self.config.gamma_min),
      bias_init=model_vdm.constant_init(self.config.gamma_min))
    self.min_gamma = self.param(
      'min_gamma', model_vdm.constant_init(self.config.gamma_min),
      (n_out,))
    self.max_minus_min_gamma = self.param(
      'max_minus_min_gamma',
      model_vdm.constant_init(self.config.gamma_max - self.config.gamma_min),
      (n_out,))

    self.l1 = nn.Dense(features=self.n_features, name='dense_1')
    self.l2 = nn.Dense(features=self.n_features, name='dense_2')
    self.l3_w = nn.Dense(
      features=n_out,
      name='dense_out_w',
      kernel_init=model_vdm.constant_init(0),
      bias_init=model_vdm.constant_init(1))
    self.l3_b = nn.Dense(
      features=n_out,
      name='dense_out_b',
      kernel_init=model_vdm.constant_init(0),
      bias_init=model_vdm.constant_init(0))
    self.l3_s = nn.Dense(
      features=n_out, name='dense_out_s',
      kernel_init=model_vdm.constant_init(0),
      bias_init=model_vdm.constant_init(1))


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
    w = nn.softplus(self.l3_w(_h))
    s = nn.softplus(self.l3_s(_h))
    b = self.l3_b(_h)
    nonlinear = s * nn.sigmoid(w * t + b)

    return self.linear(t) + nonlinear


class NoiseSchedule_polynomial(nn.Module):
  config: model_vdm.VDMConfig
  n_features: int = 32 * 32 * 3
  nonlinear: bool = True

  def setup(self):
    n_out = 32 * 32 * 3
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

    return self.min_gamma + self.max_minus_min_gamma * scaled_integral


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


class NoiseSchedule_polynomial_fixedend_stop_grad(NoiseSchedule_polynomial_fixedend):
  
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
    integral_scaling_factor = cumulative_curve_length[:, -1]
    integral_scaling_factor = jax.lax.stop_gradient(cumulative_curve_length[:, -1])
    return new_t, integral_scaling_factor


class NoiseSchedule_polynomial_fixedend_scalar(NoiseSchedule_polynomial_fixedend):

  def __call__(self, embedding, t, det_min_max=False):
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1

    if jnp.isscalar(t) or len(t.shape) == 0:
      t = t * jnp.ones((embedding.shape[0], 1))
    else:
      t = jnp.reshape(t, (-1, 1))
    assert t.shape[0] == embedding.shape[0]
    assert len(embedding.shape) == 2, f"{embedding.shape}"

    a, b, c = self._compute_coefficients(embedding)
    gamma = self._eval_polynomial(a, b, c, t)
    gamma = jnp.mean(gamma, axis=1, keepdims=True) * jnp.ones_like(gamma)
    return gamma


class NoiseSchedule_polynomial_fixedend_v2(NoiseSchedule_polynomial_fixedend):

  def setup(self):
    n_out = 32 * 32 * 3
    self.min_gamma = self.config.gamma_min
    self.max_minus_min_gamma = self.config.gamma_max - self.config.gamma_min
    self.grad_min_epsilon = 1e-3
    self.l1 = nn.Dense(features=self.n_features, name='dense_1')
    self.l2 = nn.Dense(features=self.n_features, name='dense_2')
    self.l3_a = nn.Dense(features=n_out, name='dense_out_a')
    self.l3_b = nn.Dense(features=n_out, name='dense_out_b')

  def _compute_coefficients(self, embedding):
    _h = nn.swish(self.l1(embedding))
    _h = nn.swish(self.l2(_h))
    a = self.l3_a(_h)
    b = self.l3_b(_h)
    c = jnp.ones_like(b)
    return a, b, c


class NoiseSchedule_polynomial_fixedend_nocond(nn.Module):
  config: model_vdm.VDMConfig
  n_features: int = 32 * 32 * 3
  nonlinear: bool = True

  def setup(self):
    n_out = 32 * 32 * 3
    self.min_gamma = self.config.gamma_min
    self.max_minus_min_gamma = self.config.gamma_max - self.config.gamma_min

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
    _h = nn.swish(self.l1(jnp.zeros_like(image_embedding)))
    _h = nn.swish(self.l2(_h))
    a = self.l3_a(_h)
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

    return self.min_gamma + self.max_minus_min_gamma * scaled_integral


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
  'learnable_nnet_v2': NoiseSchedule_NNet_v2,
  'learnable_nnet_time_matrix': NoiseSchedule_NNet_TimeMatrix,
  'film': NoiseSchedule_NNet_TimeMatrix_Film,
  'filmv2': NoiseSchedule_NNet_TimeMatrix_FilmV2,
  'sigmoid': NoiseSchedule_Sigmoid,
  'sigmoid_2': NoiseSchedule_Sigmoid2,
  'poly': NoiseSchedule_polynomial,
  'poly_fixedend': NoiseSchedule_polynomial_fixedend,
  'poly_fixedend_v2': NoiseSchedule_polynomial_fixedend_v2,
  'poly_fixedend_nocond': NoiseSchedule_polynomial_fixedend_nocond,
  'poly_fixedend_scalar': NoiseSchedule_polynomial_fixedend_scalar,
  'poly_fixedend_stop_grad': NoiseSchedule_polynomial_fixedend_stop_grad,
  }

