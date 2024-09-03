from functools import partial
from typing import Optional


from flax import linen as nn
from jax import numpy as jnp
from ldm import model_vdm


class ResnetBlock(nn.Module):
  '''Convolutional residual block with two convs.'''
  config: model_vdm.VDMConfig
  out_ch: Optional[int] = None
  use_grad_checkpointing = False

  @model_vdm.conditional_decorator(partial(nn.remat, static_argnums=(3,)), use_grad_checkpointing)
  @nn.compact
  def __call__(self, x, cond, deterministic: bool, enc=None):
    config = self.config

    nonlinearity = nn.swish
    normalize1 = nn.normalization.GroupNorm()
    normalize2 = nn.normalization.GroupNorm()

    if enc is not None:
      x = jnp.concatenate([x, enc], axis=-1)

    B, _, _, C = x.shape  # pylint: disable=invalid-name
    out_ch = C if self.out_ch is None else self.out_ch

    h = x
    h = nonlinearity(normalize1(h))
    h = nn.Conv(
        features=out_ch, kernel_size=(3, 3), strides=(1, 1), name='conv1')(h)

    # add in conditioning
    if cond is not None:
      conditional_bias = nn.Dense(
          features=out_ch,
          use_bias=False,
          kernel_init=nn.initializers.zeros,
          name='cond_proj')(cond)
      assert h.shape == conditional_bias.shape, f'h: {h.shape} conditional_bias:{ conditional_bias.shape}'
      assert cond.shape[0] == B
      h += conditional_bias

    h = nonlinearity(normalize2(h))
    h = nn.Dropout(rate=config.sm_pdrop)(h, deterministic=deterministic)
    h = nn.Conv(
        features=out_ch,
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv2')(h)

    if C != out_ch:
      x = nn.Dense(features=out_ch, name='nin_shortcut')(x)

    assert x.shape == h.shape
    x = x + h
    return x, x


class UNet(nn.Module):
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

    lb = config.gamma_min
    ub = config.gamma_max
    t = (g_t - lb) / (ub - lb)  # ---> [0,1]

    assert t.shape == z.shape, f'{t.shape}'
    temb = model_vdm.get_timestep_embedding(t.reshape(-1), n_embd)
    temb = temb.reshape(z.shape[0], z.shape[1], z.shape[2], 3 * n_embd)
    conditioning = conditioning[:, None, None, :] * jnp.ones(
      (z.shape[0], z.shape[1], z.shape[2], conditioning.shape[1]),
      dtype=conditioning.dtype)
    cond = jnp.concatenate([temb, conditioning], axis=-1)
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense0')(cond))
    cond = nn.swish(nn.Dense(features=n_embd * 4, name='dense1')(cond))

    # Concatenate Fourier features to input
    if config.with_fourier_features:
      z_f = model_vdm.Base2FourierFeatures(start=6, stop=8, step=1)(z)
      h = jnp.concatenate([z, z_f], axis=-1)
    else:
      h = z

    # Linear projection of input
    h = nn.Conv(features=n_embd,
                kernel_size=(3, 3),
                strides=(1, 1),
                name='conv_in')(h)
    hs = [h]

    # Downsampling
    for i_block in range(n_layers):
      block = ResnetBlock(config, out_ch=n_embd, name=f'down.block_{i_block}')
      h = block(hs[-1], cond, deterministic)[0]
      if config.with_attention:
        h = model_vdm.AttnBlock(num_heads=1, name=f'down.attn_{i_block}')(h)
      hs.append(h)

    # Middle
    h = hs[-1]
    h = ResnetBlock(config, name='mid.block_1')(h, cond, deterministic)[0]
    h = model_vdm.AttnBlock(num_heads=1, name='mid.attn_1')(h)
    h = ResnetBlock(config, name='mid.block_2')(h, cond, deterministic)[0]

    # Upsampling
    for i_block in range(n_layers + 1):
      b = ResnetBlock(config, out_ch=n_embd, name=f'up.block_{i_block}')
      h = b(jnp.concatenate([h, hs.pop()], axis=-1), cond, deterministic)[0]
      if config.with_attention:
        h = model_vdm.AttnBlock(num_heads=1, name=f'up.attn_{i_block}')(h)

    assert not hs

    # Predict noise
    normalize = nn.normalization.GroupNorm()
    h = nn.swish(normalize(h))
    eps_pred = nn.Conv(
        features=z.shape[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        kernel_init=nn.initializers.zeros,
        name='conv_out')(h)

    # Base measure
    eps_pred += z

    return eps_pred