import os  # nopep8
import functools
import time

import jax

import ldm.experiment_vdm
import ldm.dataset

from clu import checkpoint
import flax.jax_utils as flax_utils
import flax
import jax.numpy as jnp
import ldm.utils as utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML
from sklearn.decomposition import PCA
import scipy
import skimage
import collections
from sklearn.manifold import TSNE

from scipy import integrate


class Experiment_Colab(ldm.experiment_vdm.Experiment_VDM):
    def __init__(self, config, checkpoint_dir, checkpoint_num=None):
        super().__init__(config)
        ckpt = checkpoint.Checkpoint(checkpoint_dir)
        if checkpoint_num is None:
            state_dict = ckpt.restore_dict()
        else:
            state_dict = ckpt.restore_dict(
                os.path.join(checkpoint_dir, f'ckpt-{checkpoint_num}'))
        self.orig_params = flax.core.FrozenDict(state_dict['ema_params'])
        # Distribute training.
        self.params = flax_utils.replicate(self.orig_params)
        
        self.rng, sample_rng = jax.random.split(self.rng)
        self.rngs = {'sample': sample_rng}
        self.conditional_samples = functools.partial(
            self._conditional_samples,
            rng=self.rng,
            dummy_inputs=next(self.eval_iter)['images'][0])
        self.conditional_samples = utils.dist(
            self.conditional_samples, accumulate='concat', axis_name='batch')
        self.random_samples = functools.partial(
            self._random_samples,
            rng=self.rng,
            dummy_inputs=next(self.eval_iter)['images'][0])
        self.random_samples = utils.dist(
            self.random_samples, accumulate='concat', axis_name='batch')
            
    def _random_samples(self, *, dummy_inputs, rng, params):
        rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
        batch_size = dummy_inputs.shape[0]
        # sample z_0 from the diffusion model
        rng, sample_rng = jax.random.split(rng)

        T = 1000
        rng, embeddings_rng = jax.random.split(rng)

        z_init = jax.random.normal(sample_rng, (batch_size, 32, 32, 3))
        def body_fn(i, z_t):
            return self.state.apply_fn(
                variables={'params': params},
                i=i,
                T=T,
                z_t=z_t,
                embedding=logits_to_embeddings(jax.random.normal(embeddings_rng, (batch_size, 50))),
                conditioning=jnp.zeros((batch_size,), dtype='uint8'),
                rng=rng,
                method=self.model.conditional_sample,
            )

        z_0 = jax.lax.fori_loop(
            lower=0, upper=T, body_fun=body_fn, init_val=z_init)

        samples = self.state.apply_fn(
            variables={'params': params},
            z_0=z_0,
            method=self.model.generate_x,
        )

        return samples


    def _conditional_samples(self, embedding, dummy_inputs, rng, params):
        rng = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

        T = 1000
        jax.debug.print('embedding {x}', x=embedding)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] == 50
        embedding = embedding * jnp.ones((dummy_inputs.shape[0], 50))
        conditioning = jnp.zeros((dummy_inputs.shape[0],), dtype='uint8')

        # sample z_0 from the diffusion model
        rng, sample_rng = jax.random.split(rng)
        z_init = jax.random.normal(sample_rng, (dummy_inputs.shape[0], 32, 32, 3))
        def body_fn(i, z_t):
            return self.state.apply_fn(
                variables={'params': params},
                i=i,
                T=T,
                z_t=z_t,
                embedding=embedding,
                conditioning=conditioning,
                rng=rng,
                method=self.model.conditional_sample,
            )

        z_0 = jax.lax.fori_loop(
            lower=0, upper=T, body_fun=body_fn, init_val=z_init)

        samples = self.state.apply_fn(
            variables={'params': params},
            z_0=z_0,
            method=self.model.generate_x,
        )

        return samples

    def sample_conditionally(self, embedding):
        embedding = jax.tree_map(jnp.asarray, embedding)
        samples = self.conditional_samples(
            embedding=embedding, params=self.params)
        return utils.generate_image_grids(samples).astype(np.uint8)

    def sample_randomly(self):
        samples = self.random_samples(
            embedding=jax.tree_map(jnp.asarray, get_embedding()),
            params=self.params)
        return utils.generate_image_grids(samples).astype(np.uint8)

    def test(self, loader):
        """Perform one evaluation."""
        eval_metrics = []

        eval_step = 0
        for batch in loader:
            batch = jax.tree_map(jnp.asarray, batch)
            metrics = self.p_eval_step(
                self.params, batch, flax_utils.replicate(eval_step))
            eval_metrics.append(metrics['scalars'])
            # jax.debug.print('{x}', x=metrics['scalars']['eval_bpd'])
            eval_step += 1

        # average over eval metrics
        eval_metrics = utils.get_metrics(eval_metrics)
        eval_metrics = jax.tree_map(jnp.mean, eval_metrics)
        return eval_metrics


def eval_bpd_sparse_sampling(experiment, config):
    batch_size = config.training.batch_size_eval
    loader = ldm.dataset.create_one_time_eval_dataset(config, batch_size=batch_size)
    rng = jax.random.PRNGKey(0)
    bpds = []
    eval_step = 0
    for batch in loader:
        batch = jax.tree_map(jnp.asarray, batch)
        batch['images'] = jnp.squeeze(batch['images'], axis=0)
        assert batch['images'].shape == (batch_size, 32, 32, 3)
        bpd, _ = experiment.loss_fn(experiment.orig_params, batch, eval_step, rng=rng, is_train=False)
        bpds.append(bpd)
        eval_step += 1
        if eval_step % 100 == 0:
            print(f'eval_step {eval_step} cum_avg_bpd {np.mean(bpds)} ')
    print('Num eval steps:', eval_step)
    return np.mean(bpds)


def eval_bpd_dense_sampling(experiment, config, n_timesteps=128):
    loader = ldm.dataset.create_one_time_eval_dataset(config, batch_size=1)
    rng = jax.random.PRNGKey(0)
    bpds = []
    eval_step = 0
    for batch in loader:
        batch = jax.tree_map(jnp.asarray, batch)
        batch['images'] = jnp.tile(jnp.squeeze(batch['images'], axis=0), (n_timesteps, 1, 1, 1))
        assert batch['images'].shape == (n_timesteps, 32, 32, 3)
        bpd, _ = experiment.loss_fn(experiment.orig_params, batch, eval_step, rng=rng, is_train=False)
        bpds.append(bpd)
        eval_step += 1
        if eval_step % 100 == 0:
            print(f'eval_step {eval_step} cum_avg_bpd {np.mean(bpds)} ')
    print('Num eval steps:', eval_step)
    return np.mean(bpds)


def _to_flattened_numpy(x):
  """Flatten a JAX array `x` and convert it to numpy."""
  return np.asarray(x, dtype=np.float64).reshape(-1)


def _from_flattened_numpy(x, shape):
  """Form a JAX array with the given `shape` from a flattened numpy array `x`."""
  return jnp.asarray(x, dtype=jnp.float32).reshape(* shape)


def _get_value_div_fn(fn):
  """Return both the function value and its estimated divergence via Hutchinson's trace estimator."""

  def value_div_fn(x, embeddings, t, hutchinson_noise):
    def value_grad_fn(data, e):
      f = fn(data, e, t)
      return jnp.sum(f * hutchinson_noise), f
    grad_fn_eps, value = jax.grad(
      value_grad_fn, has_aux=True, argnums=0)(x, embeddings)
    assert grad_fn_eps.shape == hutchinson_noise.shape
    return value, jnp.sum(grad_fn_eps * hutchinson_noise, axis=tuple(range(1, len(x.shape))))

  return value_div_fn


def _prior_logp(z):
  N = np.prod(z.shape[1:])
  return -0.5 * N * jnp.log(
    2 * np.pi) - 0.5 * jnp.sum(z ** 2, axis=(1, 2, 3))

def _gumbel_kl_loss(logits):
  assert logits.shape[-1] == 50
  q_z = jax.nn.softmax(logits)
  log_q_z = jax.nn.log_softmax(logits)
  return jnp.sum(
    q_z * (log_q_z - jnp.log(1.0 / logits.shape[-1])),
    axis=-1)


class Hutchinson:
  def __init__(self, hutchinson_type, shape, rng, deterministic=False):
    self.rng = rng
    self.hutchinson_type = hutchinson_type
    self.shape = shape
    self.deterministic = deterministic
    if deterministic:
      self.det_noise = self._deterministic_noise()

  def noise(self):
    if self.deterministic:
        return self.det_noise
    else:
       return self._sample_noise()

  def _sample_noise(self):
    self.rng, noise_rng = jax.random.split(self.rng)
    if self.hutchinson_type == 'Gaussian':
      return jax.random.normal(noise_rng, self.shape)
    elif self.hutchinson_type == 'Rademacher':
      return -1 + 2 * jax.random.randint(
        noise_rng, self.shape, minval=0, maxval=2).astype(jnp.float32)

  def _deterministic_noise(self):
    if self.hutchinson_type == 'Gaussian':
      return jax.random.normal(self.rng, self.shape)
    elif self.hutchinson_type == 'Rademacher':
      return jax.random.randint(
        self.rng, self.shape, minval=0, maxval=2).astype(jnp.float32) * 2 - 1


def get_ode_likelihood_fn(
    experiment, hutchinson_type='Rademacher', rtol=1e-5, atol=1e-5, method='RK45',
    dequantization='uniform', high_precision=False):
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    
  Returns:
    A function that takes random states, replicated training states, and a batch of data points
      and returns the log-likelihoods in bits/dim, the latent code, and the number of function
      evaluations cost by computation.
  """
  def _value_div_fn(x, embeddings, t, hutchinson_noise):
    """Pmapped divergence of the drift function."""
    def _reverse_ode(xt, embeds, t):
      return experiment.state.apply_fn(
          variables={'params': experiment.orig_params},
          xt=xt,
          embeddings=embeds,
          t=t,
          high_precision=high_precision,
          method=experiment.model.reverse_ode,
      )
    value_div_fn = _get_value_div_fn(_reverse_ode)
    return value_div_fn(x, embeddings, t, hutchinson_noise)

  p_value_div_fn = jax.pmap(_value_div_fn, in_axes=(0, 0, None, 0))
  p_prior_logp_fn = jax.pmap(_prior_logp)  # Pmapped log-PDF of the SDE's prior distribution

  @jax.pmap
  def _compute_logits(data):
    return experiment.state.apply_fn(
        variables={'params': experiment.orig_params},
        images_int=data,
        method=experiment.model.apply_encoder,
    )

  def likelihood_fn(rng, data, deterministic_noise=False):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Returns:
      bpd: A JAX array of shape [#devices, batch size]. The log-likelihoods on `data` in bits/dim.
      z: A JAX array of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    rng, init_noise_rng = jax.random.split(rng)
    shape = data.shape
    assert len(shape) == 5
    # step_rng, epsilon = _get_hutchinson_noise(hutchinson_type, shape, step_rng)
    data = 2 * ((data.round() + .5) / 256) - 1

    if dequantization == 'uniform':
      u = jax.random.uniform(init_noise_rng, data.shape) - 0.5
      u = 2 * u / 256
      log_q_eps = None
    elif dequantization == 'tn':
      gt = -13.3
      u = jax.random.truncated_normal(
        key=init_noise_rng,
        upper=3,
        lower=-3,
        shape=data.shape,
      )
      # u = jax.random.normal(init_noise_rng, data.shape)
      # In Eqn.(28) in https://openreview.net/pdf?id=jVR2fF8x8x Z = 0.9974613
      log_q_eps = p_prior_logp_fn(u) - (32 * 32 * 3) * jnp.log(0.9974613)
      u = u * jnp.exp(0.5 * gt)
    else:
      assert False
    data = data + u
    logits = _compute_logits(jnp.clip(128 * (data + 1) - 0.5, a_min=0, a_max=255).round())
    auxiliary_latent_loss = _gumbel_kl_loss(logits)
    embeddings = logits_to_embeddings(logits.reshape(-1, logits.shape[-1])).reshape(* logits.shape)
    assert embeddings.ndim == 3

    rng, hutchinson_rng = jax.random.split(rng)
    hutchinson = Hutchinson(
      hutchinson_type, shape, hutchinson_rng, deterministic=deterministic_noise)

    def ode_func(t, x):
      xt = _from_flattened_numpy(x[:-shape[0] * shape[1]], shape)
      epsilon = hutchinson.noise()
      drift, logp_grad = p_value_div_fn(xt, embeddings, t, epsilon)
      drift = _to_flattened_numpy(drift)
      logp_grad = _to_flattened_numpy(logp_grad)
      new_x = np.concatenate([drift, logp_grad], axis=0)
      assert new_x.shape == x.shape, f'{new_x.shape} != {x.shape}'
      return new_x

    init = jnp.concatenate([_to_flattened_numpy(data), np.zeros((shape[0] * shape[1],))], axis=0)
    solution = integrate.solve_ivp(ode_func, (0, 1), init, rtol=rtol, atol=atol, method=method)
    # nfe = solution.nfev
    # print(rtol, atol)
    zp = jnp.asarray(solution.y[:, -1])
    z = _from_flattened_numpy(zp[:-shape[0] * shape[1]], shape)
    delta_logp = zp[-shape[0] * shape[1]:].reshape((shape[0], shape[1]))
    prior_logp = p_prior_logp_fn(z)
    assert prior_logp.shape == delta_logp.shape == auxiliary_latent_loss.shape
    log_p = prior_logp + delta_logp

    return log_p.reshape(-1), log_q_eps.reshape(-1), auxiliary_latent_loss.reshape(-1)

  return likelihood_fn


def get_sample_fn(
    experiment, hutchinson_type='Rademacher', rtol=1e-5, atol=1e-5, method='RK45',
    high_precision=False):
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    
  Returns:
    A function that takes random states, replicated training states, and a batch of data points
      and returns the log-likelihoods in bits/dim, the latent code, and the number of function
      evaluations cost by computation.
  """
  def _value_div_fn(x, embeddings, t, hutchinson_noise):
    """Pmapped divergence of the drift function."""
    def _reverse_ode(xt, embeds, t):
      return experiment.state.apply_fn(
          variables={'params': experiment.orig_params},
          xt=xt,
          embeddings=embeds,
          t=t,
          high_precision=high_precision,
          method=experiment.model.reverse_ode,
      )
    value_div_fn = _get_value_div_fn(_reverse_ode)
    return value_div_fn(x, embeddings, t, hutchinson_noise)

  p_value_div_fn = jax.pmap(_value_div_fn, in_axes=(0, 0, None, 0))

  def sample_fn(rng, deterministic_noise=False, sample_size=32):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Returns:
      bpd: A JAX array of shape [#devices, batch size]. The log-likelihoods on `data` in bits/dim.
      z: A JAX array of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    rng, logits_rng = jax.random.split(rng)
    embeddings = logits_to_embeddings(jax.random.normal(logits_rng, (sample_size, 50)))
    
    n_devices =  jax.local_device_count()
    shape = (n_devices, sample_size // n_devices, 32, 32, 3)
    embeddings = embeddings.reshape(n_devices, sample_size // n_devices, 50)

    rng, hutchinson_rng = jax.random.split(rng)
    hutchinson = Hutchinson(
      hutchinson_type, shape, hutchinson_rng, deterministic=deterministic_noise)

    def ode_func(t, x):
      xt = _from_flattened_numpy(x, shape)
      drift, _ = p_value_div_fn(xt, embeddings, t, hutchinson.noise())
      new_x = _to_flattened_numpy(drift)
      assert new_x.shape == x.shape, f'{new_x.shape} != {x.shape}'
      return new_x
    rng, prior_rng = jax.random.split(rng)
    prior_sample = jax.random.normal(prior_rng, shape)
    init = _to_flattened_numpy(prior_sample)
    solution = integrate.solve_ivp(ode_func, (1, 0), init, rtol=rtol, atol=atol, method=method)
    z = _from_flattened_numpy(jnp.asarray(solution.y[:, -1]), shape)
    return z, solution.nfev

  return sample_fn


def _get_bpd_offset(dequantization, num_is):
  if dequantization == 'uniform':
    offset = jnp.log2(128)
  elif dequantization == 'tn':
    gt = - 13.3
    log_sigma = 0.5 * (gt - jax.nn.softplus(gt))
    extra_terms = 0.0
    if num_is == 1:
      extra_terms = 0.5 * (1 + jnp.log(2 * jnp.pi)) - 0.01522
    return - (extra_terms + log_sigma) / jnp.log(2)
  else:
    assert False
  return offset


def eval_bpd_ode(
      experiment, config, deterministic_noise,
      hutchinson_type, dequantization='tn', num_is=1, num_iters=1,
      rtol=1e-5, atol=1e-5):
  bpd_means = []
  rng = jax.random.PRNGKey(0)
  for i in range(num_iters):
    rng, iter_rng = jax.random.split(rng)
    mean, _ = _eval_bpd_ode(
      experiment=experiment,
      config=config,
      rng=iter_rng,
      deterministic_noise=deterministic_noise,
      hutchinson_type=hutchinson_type,
      dequantization=dequantization,
      num_is=num_is,
      rtol=rtol,
      atol=atol)
    print(f'[Iter {i}] Test BPD:{mean}')
    bpd_means.append(mean)
  return np.mean(bpd_means)


def _eval_bpd_ode(experiment, config, rng, deterministic_noise,
                  hutchinson_type, dequantization='tn', num_is=1,
                  rtol=1e-5, atol=1e-5):
  batch_size = config.training.batch_size_eval
  loader = ldm.dataset.create_one_time_eval_dataset(config, batch_size=batch_size)
  bpds = []
  eval_step = 0
  likelihood_function = get_ode_likelihood_fn(
    experiment,
    rtol=rtol,
    atol=atol,
    hutchinson_type=hutchinson_type,
    dequantization=dequantization)
  bpd_offset = _get_bpd_offset(dequantization, num_is)
  for batch in loader:
    data = jax.tree_map(jnp.asarray, batch)['images']

    log_ps = []
    log_qs = []
    aux_losses = []
    start_time = time.time()
    for _ in range(num_is):
      rng, likelihood_rng = jax.random.split(rng)
      log_p, log_q_eps, aux_loss = likelihood_function(
        likelihood_rng, data, deterministic_noise=deterministic_noise)
      log_ps.append(log_p)
      log_qs.append(log_q_eps)
      aux_losses.append(aux_loss)
    # print(np.mean(aux_losses), np.mean(aux_loss))
    log_ps = jnp.asarray(log_ps)
    log_qs = jnp.asarray(log_qs)
    if num_is == 1:
      iws = log_ps
    else:
      # print(log_ps.shape, log_qs.shape)
      iws = jax.scipy.special.logsumexp(log_ps - log_qs, axis=0) - jnp.log(num_is)
      assert log_ps.shape == (num_is, iws.shape[0])
      # print(iws.shape)
    # all aux_loss in aux_losses are the same since the inputs are the same.
    bpd = jnp.mean(- iws + aux_loss) / (32 * 32 * 3 * jnp.log(2)) + bpd_offset
    bpds.append(bpd)

    print('Eval step:{}\tcum. bpd: {:.3f}: {:.2f} mins'.format(
      eval_step, np.mean(bpds), (time.time() - start_time) / 60))
    eval_step += 1

  print('Num eval steps:', eval_step)
  return np.mean(bpds)


def get_logits(experiment, num_batches=30):
    logits = []
    images = []
    for _ in range(num_batches):
        batch = experiment.eval_iter.next()
        batch = jax.tree_map(jnp.asarray, batch)
        logits.append(experiment.state.apply_fn(
            variables={'params': experiment.orig_params},
            images_int=batch['images'][0].reshape(-1, 32, 32, 3),
            method=experiment.model.apply_encoder,
        ))
        images.append(batch['images'][0])
    return jax.numpy.concatenate(logits), jax.numpy.concatenate(images)

def logits_to_embeddings(logits):
    top_k_vals, _ = jax.lax.top_k(logits, 15)
    assert top_k_vals.shape == (logits.shape[0], 15)
    return (logits >= top_k_vals[:, -1][:, None]).astype(float)


def noise_schedule_per_embedding(experiment, embeddings, time_steps=None):
    if time_steps is None:
        time_steps = jnp.linspace(0, 1, 128)
    noise_schedules = []
    for i in range(embeddings.shape[0]):
        noise_schedules.append(
            experiment.state.apply_fn(
                variables={'params': experiment.orig_params},
                embedding=jnp.repeat(
                    embeddings[i:i+1], 128, axis=0),
                t=time_steps,
                method=experiment.model._get_gamma,
            )
        )
    return noise_schedules


def plot_noise_schedule(noise_schedules, epoch=''):
    plt.figure()
    plt.plot(noise_schedules[0])
    plt.title(f'Noise Schedule per pixel for an input epoch:{epoch}')
    plt.xticks(
        (np.linspace(0, 1, 10) * len(noise_schedules[0])).astype(int),
        ['{:.1f}'.format(i) for i in np.linspace(0, 1, 10)])
    plt.ylabel('$\gamma(t)$')
    plt.xlabel('$t$')


def get_embedding(batch_size=2, latent_size=50, shift=0):
    ones = jnp.ones((batch_size, 15))
    zeros = jnp.zeros((batch_size, latent_size - 15))
    return jnp.roll(jnp.concatenate([ones, zeros], axis=1),
                    shift=shift, axis=1)

def plot_sequence_images(image_array, dpi=100.0):
    ''' Display images sequence as an animation in jupyter notebook
    
    Args:
        image_array(numpy.ndarray): image_array.shape equal to (num_images, height, width, num_channels)
    '''
    fig = plt.figure(
        figsize=(image_array[0].shape[1] / dpi,
                 image_array[0].shape[0] / dpi),
        dpi=dpi)
    im = plt.figimage(image_array[0])

    def animate(i):
        im.set_array(image_array[i])
        return (im,)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(image_array), interval=800,
        repeat_delay=1, repeat=True)
    display(HTML(anim.to_html5_video()))


def animate_noise_schedule(noise_schedules, dpi=100.0):
    ''' Display images sequence as an animation in jupyter notebook
    
    Args:
        image_array(numpy.ndarray): image_array.shape equal to (num_images, height, width, num_channels)
    '''
    fig, ax = plt.subplots()

    def animate(i):
        noise_schedule = noise_schedules[i]
        ax.clear()
        ax.set_title(f'{3 * 10 * (i + 1)} k / 500k steps')
        ax.plot(noise_schedule)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(noise_schedules), interval=800,
        repeat_delay=1, repeat=True)
    display(HTML(anim.to_html5_video()))


def plot_heat_map(noise_schedules, count=3):
    for ns in noise_schedules[:count]:
        fig = plt.figure(figsize=(6, 6))
        num_cols = 10
        # ns = (ns - jnp.min(ns)) / (jnp.max(ns) - jnp.min(ns))
        for t in range(num_cols):
            fig.add_subplot(1, num_cols, t + 1)
            timestep = int(ns.shape[0] * t / num_cols)
            nspp = ns[timestep]
            nspp = nspp.reshape((32, 32, 3))
            nspp = nspp[2:-2, 2:-2, :]
            nspp = (nspp - jnp.min(nspp)) / (jnp.max(nspp) - jnp.min(nspp))
            nspp = skimage.color.rgb2gray(nspp)
            # nspp = np.mean(nspp, axis=-1)
            # print(nspp)
            plt.imshow(
                nspp,
                cmap='hot',
                interpolation='nearest')
            plt.title('t={:.1f}'.format(t / num_cols), fontsize=8)
            plt.xticks([], [])
            plt.yticks([], [])
        # plt.colorbar()


def plot_histogram(noise_schedules, count=3):
  for ns in noise_schedules[:count]:
    num_cols = 5
    fig = plt.figure(figsize=(num_cols, 1))
    ns = (ns - jnp.min(ns)) / (jnp.max(ns) - jnp.min(ns))
    for t in range(num_cols):
        fig.add_subplot(1, num_cols, t + 1)
        plt.hist(
            ns[int(ns.shape[0] * t / num_cols)],
            bins=100)
        plt.xticks([])
        plt.yticks([])


class Clustering:

    def __init__(self, images, logits, embeddings, noise_schedules, threshold=0.8) -> None:
        self.images = images
        # self.embeddings = embeddings
        # self.logits = logits
        self.clusters = collections.defaultdict(list)
        self.dotp = embeddings @ embeddings.transpose()
        self.threshold = threshold * np.max(self.dotp)
        self.visited = set()
        self.cluster_count = 0
        self.noise_schedules = noise_schedules


    def print_clusters(self, cluster_count=20, cluster_size_max=10):
        indices = np.where(np.sum(self.dotp > self.threshold, axis=0) > 1)[0]
        noise_schedules_per_cluster = []
        for i in indices[:cluster_count]:
            cluster = [self.images[i]]
            noise_schedules_per_cluster.append(self.noise_schedules[i])
            for j in np.where(self.dotp[i] > self.threshold)[0]:
                # print(i, j)
                if i == j:
                    continue
                cluster.append(self.images[j])
                if len(cluster) == cluster_size_max:
                    break
            fig = plt.figure(figsize=(len(cluster), 1))
            for index in range(len(cluster)):
                fig.add_subplot(1, len(cluster), index + 1)
                plt.imshow(cluster[index])
                plt.xticks([])
                plt.yticks([])
        plt.figure()
        for ns in noise_schedules_per_cluster:
            plt.plot([np.mean(ns_pixel) for ns_pixel in ns])
            plt.xticks(
                (np.linspace(0, 1, 10) * len(ns)).astype(int),
                ['{:.1f}'.format(i) for i in np.linspace(0, 1, 10)])
            plt.ylabel('$\gamma(t)$')
            plt.xlabel('$t$')   
        plt.title('Noise schedule for an image from each cluster')


def plot_tsne_transformation(data):
    for p in range(5, 55, 10):
        tsne = TSNE(2, perplexity=p)
        tsne_embeddings = tsne.fit_transform(data)
        plt.figure()
        plt.title(f'perplexity: {p}')
        plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1])


def pca_transformation(data, n_components=4):
    pca = PCA(n_components=n_components, svd_solver='full')
    pca.fit(data)
    print('variance ratio', pca.explained_variance_ratio_)
    print('singular values', pca.singular_values_)
    return pca.fit_transform(data)


def dct2(image):
    image = skimage.color.rgb2gray(image)
    return scipy.fftpack.dct(
        scipy.fftpack.dct(image.T, norm='ortho').T, norm='ortho')


def animate_scatter(xs, ys, cs, dpi=100.0):
    ''' Display images sequence as an animation in jupyter notebook
    
    Args:
        image_array(numpy.ndarray): image_array.shape equal to (num_images, height, width, num_channels)
    '''
    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        ax.set_title(f'{3 * 10 * (i + 1)} k / 500k steps')
        ax.axis([-2, 2, -2, 2])
        ax.scatter(xs[i], ys[i], c=(cs[i] > np.mean(cs[i])))

    anim = animation.FuncAnimation(
        fig, animate, frames=len(xs), interval=800,
        repeat_delay=1, repeat=True)
    display(HTML(anim.to_html5_video()))