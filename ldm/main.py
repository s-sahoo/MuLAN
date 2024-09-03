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

import os  # nopep8
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable TF info/warnings # nopep8

import jax
import tensorflow as tf
import tensorflow_io as tfio
from absl import logging
from absl import flags
from absl import app
from ml_collections import config_flags
from ldm.utils import get_workdir
import ldm.experiment_vdm

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=False)
flags.DEFINE_string('workdir', None, 'Work unit directory.')
flags.DEFINE_string('checkpoint', '', 'Checkpoint to evaluate.')
flags.DEFINE_string('mode', 'train', 'train / eval')
flags.DEFINE_string('model', 'vdm', 'vdm')
flags.mark_flags_as_required(['config', 'workdir'])
flags.DEFINE_string('log_level', 'info', 'info/warning/error')

# jax.config.update("jax_debug_nans", True)
jax.config.update("jax_default_matmul_precision", "float32")
# jax.config.update('jax_disable_jit', True)

#@jax.default_matmul_precision("tensorfloat32")
def main(argv):
  del argv
  if jax.process_index() == 0:
    logging.set_verbosity(FLAGS.log_level)
  else:
    logging.set_verbosity('error')
  logging.warning('=== Start of main() ===')

  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX. (Not necessary with TPU.)
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d',
               jax.process_index(), jax.process_count())
  logging.info("JAX devices: %r", jax.devices())
  if FLAGS.model == 'vdm':
    experiment = ldm.experiment_vdm.Experiment_VDM(FLAGS.config)
  else:
    raise RuntimeError(f"{flags.model} is not implemented")

  if FLAGS.mode == 'train':
    workdir = os.path.join(FLAGS.workdir, get_workdir())
    logging.info('Training at workdir: '+FLAGS.workdir)
    experiment.train_and_evaluate(workdir)
  elif FLAGS.mode == 'eval':
    experiment.evaluate(FLAGS.workdir, FLAGS.checkpoint)
  else:
    raise Exception('Unknown FLAGS.mode')


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
