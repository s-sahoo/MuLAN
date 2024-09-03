import os  # nopep8
# os.chdir('learned-diffusion')
# os.chdir('../')


from absl import logging
from absl import flags
from absl import app

import jax
from ml_collections import config_flags
import tensorflow as tf

from ldm.notebook_utils import Experiment_Colab, eval_bpd_dense_sampling, eval_bpd_sparse_sampling
from ldm.notebook_utils import eval_bpd_ode

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=False)
flags.DEFINE_string('checkpoint_directory', None, 'Work unit directory.')
flags.DEFINE_string('checkpoint', None, 'Checkpoint to evaluate.')
flags.DEFINE_string('bpd_eval_method', 'ode', 'Dense / Sparse / ODE sampling to evaluate BPD.')
flags.DEFINE_string('log_level', 'info', 'info/warning/error')
flags.DEFINE_integer('n_timesteps', 128, 'discrete timesteps for dense sampling to evaluate BPD.')
flags.DEFINE_integer('n_is', 20, 'Number of Importance Samples.')
flags.DEFINE_integer('num_iters', 1, 'Number of iterations on test set.')
flags.DEFINE_bool('deterministic_noise', False, 'Number of Importance Samples.')
flags.DEFINE_string('hutchinson_type', 'Rademacher', 'Hutchinson noise type: (Rademacher/Gaussian)')
flags.DEFINE_float('rtol', 1e-5, 'rtol for the ODE solver')
flags.DEFINE_float('atol', 1e-5, 'atol for the ODE solver')
flags.mark_flags_as_required(['config', 'checkpoint_directory'])

def main(argv):
  del argv
  if jax.process_index() == 0:
    logging.set_verbosity(FLAGS.log_level)
  else:
    logging.set_verbosity('error')
  logging.warning('=== Start of main() ===')
  logging.warning(f'Num discrete timesteps: {FLAGS.n_timesteps}')

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX. (Not necessary with TPU.)
  tf.config.experimental.set_visible_devices([], 'GPU')

  jax.debug.print('loss_recon {x}', x=FLAGS)
  logging.info('JAX process: %d / %d',
               jax.process_index(), jax.process_count())
  logging.info("JAX devices: %r", jax.devices())
  ckpt_nums = []
  for i in os.listdir(FLAGS.checkpoint_directory):
    if 'ckpt' in i:
      ckpt_nums.append(int(i.split('.')[0].split('-')[1]))
  ckpt_nums = sorted(list(set(ckpt_nums)))
  print(f'Found ckpts:{ckpt_nums[0]}: {ckpt_nums[-1]}')
  print(f'rtol:{FLAGS.rtol} atol:{FLAGS.atol}')
  print('BPD eval method:', FLAGS.bpd_eval_method)
  print(FLAGS.config)
  if FLAGS.checkpoint is None:
    ckpt_num = ckpt_nums[-1]
  else:
    ckpt_num = FLAGS.checkpoint
  experiment = Experiment_Colab(FLAGS.config,
                                FLAGS.checkpoint_directory,
                                ckpt_num)
  if FLAGS.bpd_eval_method == 'sparse':
    bpd = eval_bpd_sparse_sampling(experiment, FLAGS.config)
  elif FLAGS.bpd_eval_method == 'dense':
    bpd = eval_bpd_dense_sampling(
      experiment, FLAGS.config, n_timesteps=FLAGS.n_timesteps)
  elif FLAGS.bpd_eval_method == 'ode':
    bpd = eval_bpd_ode(
      experiment, FLAGS.config,
      hutchinson_type=FLAGS.hutchinson_type,
      deterministic_noise=FLAGS.deterministic_noise,
      num_iters=FLAGS.num_iters,
      num_is=FLAGS.n_is,
      rtol=FLAGS.rtol, atol=FLAGS.atol)
  
  print(f'Test BPD:{bpd} ckpt:{ckpt_num}')


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)