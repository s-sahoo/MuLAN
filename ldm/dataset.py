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

"""Dataset loader and processor."""
from typing import Tuple
from functools import partial

from clu import deterministic_data
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

AUTOTUNE = tf.data.experimental.AUTOTUNE


def calculate_blur_radius(cutoff_frequency, image_dimensions):
    sigma = cutoff_frequency * min(image_dimensions)
    blur_radius = int(2 * (sigma + 1) / 0.3 + 1)
    blur_radius = min(blur_radius, *image_dimensions)

    if blur_radius % 2 == 0:
        blur_radius -= 1

    return sigma, blur_radius


def split_image_freqs(image , cutoff_frequency=0.06):
    # Load the image

    # Calculate the blur radius based on the cutoff frequency
    sigma, blur_radius = calculate_blur_radius(cutoff_frequency, image.shape[:2])

    channels = tf.split(image, num_or_size_splits=image.shape[-1], axis=-1)
    blurred_channels = []
    for channel in channels:
        #channel_blurred = tf.image.resize_with_crop_or_pad(tf.image.resize(channel, (blur_radius, blur_radius), method='gaussian'), *image.shape[:2])
        channel_blurred = tfa.image.gaussian_filter2d(channel, blur_radius, sigma)
        blurred_channels.append(channel_blurred)
    blurred_image = tf.concat(blurred_channels, axis=-1)
    # Subtract the blurred image from the original image to get the high pass component
    high_pass_image = tf.subtract(image, tf.math.minimum(blurred_image, image))

    # The blurred image is the low pass component
    low_pass_image = blurred_image
    assert low_pass_image.shape == high_pass_image.shape
    assert low_pass_image.shape == image.shape
    assert high_pass_image.dtype == low_pass_image.dtype == np.uint8

    return low_pass_image, high_pass_image


def create_dataset(config, data_rng):
    data_rng = jax.random.fold_in(data_rng, jax.process_index())
    rng1, rng2 = jax.random.split(data_rng)
    if config.data.dataset == 'cifar10':
      _, train_ds = create_train_dataset(
          'cifar10',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          _preprocess_cifar10)

      _, eval_ds = create_eval_dataset(
          'cifar10',
          config.training.batch_size_eval,
          'test',
          rng2,
          _preprocess_cifar10)
    elif config.data.dataset == 'cifar10_splitintensity':
      _, train_ds = create_train_dataset(
          'cifar10',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          _preprocess_cifar10_intensity_split)

      _, eval_ds = create_eval_dataset(
          'cifar10',
          config.training.batch_size_eval,
          'test',
          rng2,
          _preprocess_cifar10_intensity_split)
    elif config.data.dataset == 'cifar10_splitfreq':
      _, train_ds = create_train_dataset(
          'cifar10',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          partial(_preprocess_cifar10_freq_split, split_image=True))

      _, eval_ds = create_eval_dataset(
          'cifar10',
          config.training.batch_size_eval,
          'test',
          rng2,
          partial(_preprocess_cifar10_freq_split, split_image=True))
    elif config.data.dataset == 'cifar10_bothfreq':
      _, train_ds = create_train_dataset(
          'cifar10',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          partial(_preprocess_cifar10_freq_split, split_image=False))

      _, eval_ds = create_eval_dataset(
          'cifar10',
          config.training.batch_size_eval,
          'test',
          rng2,
          partial(_preprocess_cifar10_freq_split, split_image=False))
      
    elif config.data.dataset == 'cifar10_aug':
      _, train_ds = create_train_dataset(
          'cifar10',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          _preprocess_cifar10_augment)

      _, eval_ds = create_eval_dataset(
          'cifar10',
          config.training.batch_size_eval,
          'test',
          rng2,
          _preprocess_cifar10)
    elif config.data.dataset == "dtd_shapes3d":
      train_dtd_info, train_dtd = create_train_dataset(
          'dtd',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          partial(_preprocess_cifar10, resize=32))
      eval_dtd_info, eval_dtd = create_eval_dataset(
          'dtd',
          config.training.batch_size_eval,
          'test',
          rng2,
          partial(_preprocess_cifar10, resize=32))
      train_shapes3d_info, train_shapes3d = create_train_dataset(
          'shapes3d',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          partial(_preprocess_cifar10, resize=32, label_key="label_shape"))
      eval_shapes3d_info, eval_shapes3d = create_eval_dataset(
          'shapes3d',
          config.training.batch_size_eval,
          'train', # TODO FIX SHAPES3D DOES NOT HAVE A TEST SET
          rng2,
          partial(_preprocess_cifar10, resize=32, label_key="label_shape"))
      shape_labels_num = train_shapes3d_info.features['label_shape'].num_classes
      def update_labels(example):
        example['labels'] = example['labels'] + shape_labels_num
        return example
      train_dtd = train_dtd.map(update_labels)
      eval_dtd = eval_dtd.map(update_labels)
      train_ds = tf.data.Dataset.sample_from_datasets([train_dtd, train_shapes3d])
      eval_ds = tf.data.Dataset.sample_from_datasets([eval_dtd, eval_shapes3d])
    elif config.data.dataset == 'cifar10_aug_with_channel':
      _, train_ds = create_train_dataset(
          'cifar10',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          _preprocess_cifar10_augment_with_channel_flip)

      _, eval_ds = create_eval_dataset(
          'cifar10',
          config.training.batch_size_eval,
          'test',
          rng2,
          _preprocess_cifar10)

    elif config.data.dataset == 'imagenet32':
      _, train_ds = create_train_dataset(
          'downsampled_imagenet/32x32',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          partial(_preprocess_cifar10, label_key=None))

      _, eval_ds = create_eval_dataset(
          'downsampled_imagenet/32x32',
          config.training.batch_size_eval,
          'validation',
          rng2,
          partial(_preprocess_cifar10, label_key=None))
    elif config.data.dataset == 'imagenet32r':
      _, train_ds = create_train_dataset(
          'imagenet_resized/32x32',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          _preprocess_cifar10)

      _, eval_ds = create_eval_dataset(
          'imagenet_resized/32x32',
          config.training.batch_size_eval,
          'validation',
          rng2,
          _preprocess_cifar10)
    elif config.data.dataset == 'imagenet64':
      _, train_ds = create_train_dataset(
          'downsampled_imagenet/64x64',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          partial(_preprocess_cifar10, label_key=None))

      _, eval_ds = create_eval_dataset(
          'downsampled_imagenet/64x64',
          config.training.batch_size_eval,
          'validation',
          rng2,
          partial(_preprocess_cifar10, label_key=None))
    elif config.data.dataset == "fashion_mnist32":
      _, train_ds = create_train_dataset(
          'fashion_mnist',
          config.training.batch_size_train,
          config.training.substeps,
          rng1,
          _preprocess_fmnist)

      _, eval_ds = create_eval_dataset(
          'fashion_mnist',
          config.training.batch_size_eval,
          'test',
          rng2,
          _preprocess_fmnist)
    else:
      raise Exception("Unrecognized config.data.dataset")

    return iter(train_ds), iter(eval_ds)

def create_train_dataset(
        task: str,
        batch_size: int,
        substeps: int,
        data_rng,
        preprocess_fn) -> Tuple[tfds.core.DatasetInfo, tf.data.Dataset]:
  """Create datasets for training."""
  # Compute batch size per device from global batch size..
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must be divisible by "
                     f"the number of devices ({jax.device_count()}).")
  per_device_batch_size = batch_size // jax.device_count()

  dataset_builder = tfds.builder(task)
  dataset_builder.download_and_prepare()

  train_split = deterministic_data.get_read_instruction_for_host(
      "train", dataset_builder.info.splits["train"].num_examples)
  batch_dims = [jax.local_device_count(), substeps, per_device_batch_size]

  train_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=train_split,
      num_epochs=None,
      shuffle=True,
      batch_dims=batch_dims,
      preprocess_fn=preprocess_fn,
      prefetch_size=tf.data.experimental.AUTOTUNE,
      rng=data_rng)

  return dataset_builder.info, train_ds


def create_eval_dataset(
        task: str,
        batch_size: int,
        subset: str,
        data_rng,
        preprocess_fn) -> Tuple[tfds.core.DatasetInfo, tf.data.Dataset]:
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must be divisible by "
                     f"the number of devices ({jax.device_count()}).")
  per_device_batch_size = batch_size // jax.device_count()

  dataset_builder = tfds.builder(task)

  eval_split = deterministic_data.get_read_instruction_for_host(
      subset, dataset_builder.info.splits[subset].num_examples)
  batch_dims = [jax.local_device_count(), per_device_batch_size]

  eval_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=eval_split,
      num_epochs=None,
      shuffle=True,
      batch_dims=batch_dims,
      preprocess_fn=preprocess_fn,
      prefetch_size=tf.data.experimental.AUTOTUNE,
      rng=data_rng)

  return dataset_builder.info, eval_ds

def _preprocess_cifar10(features, resize=None, label_key="label"):
  """Helper to extract images from dict."""
  conditioning = tf.zeros((), dtype='uint8')
  image = features["image"]
  if resize is not None:
    assert isinstance(resize, int)
    image = tf.image.resize(image, (resize, resize), antialias=True)
    image = tf.cast(image, 'uint8')
  if label_key is None:
      label = 0
  else:
      label = features[label_key]
  return {"images": image, "labels": label, "conditioning": conditioning}

def _preprocess_cifar10_freq_split(features, split_image=True):
  features = _preprocess_cifar10(features)
  low, high = split_image_freqs(features["images"])
  if split_image:
    features["images"] = tf.concat([low[:16, :, :], high[16:, :, :]], axis=0)
  else:
    features["images"] = tf.concat([low, high], axis=0)
  return features

def _preprocess_cifar10_intensity_split(features):
  features = _preprocess_cifar10(features)
  if features['labels'] < 5:
    features["images"] = tf.concat([127 + 0 * features["images"][:16, :, :],
                                    features["images"][16:, :, :]], axis=0)
    features['images'] = tf.cast(features["images"], 'uint8')
  else:
    features["images"] = tf.concat([features["images"][:16, :, :],
                                    127 + 0 * features["images"][16:, :, :]], axis=0)
    features['images'] = tf.cast(features["images"], 'uint8')
  return features

def _preprocess_fmnist(features, resize=32):
  conditioning = tf.zeros((), dtype='uint8')
  image = features["image"]
  #image = image.reshape(-1, 28, 28, 1)
  if resize is not None:
    assert isinstance(resize, int)
    image = tf.image.resize(image, (resize, resize), antialias=True)#, method=tf.image.ResizeMethod.lanczos3)
  image = tf.repeat(image, repeats=3, axis=-1) # convert to 3dim
  image = tf.cast(image, 'uint8')

  # TODO FIX THE MODEL SO IT CAN TAKE GRAYSCALE IMAGES
  return {"images": image, "labels": features["label"], "conditioning": conditioning}

def _preprocess_cifar10_augment(features):
  img = features['image']
  img = tf.cast(img, 'float32')

  # random left/right flip
  _img = tf.image.flip_left_right(img)
  aug = tf.random.uniform(shape=[]) > 0.5
  img = tf.where(aug, _img, img)

  # random 90 degree rotations
  u = tf.random.uniform(shape=[])
  k = tf.cast(tf.math.ceil(3. * u), tf.int32)
  _img = tf.image.rot90(img, k=k)
  _aug = tf.random.uniform(shape=[]) > 0.5
  img = tf.where(_aug, _img, img)
  aug = aug | _aug
  aug = tf.cast(aug, 'uint8')

  return {'images': img, "labels": features["label"], 'conditioning': aug}


def create_one_time_eval_dataset(config, num_epochs=1, batch_size=None) -> tf.data.Dataset:
  if batch_size is None:
    batch_size = config.training.batch_size_eval
  if config.data.dataset == 'cifar10':
    task = 'cifar10'
    subset = 'test'
    preprocess_fn = _preprocess_cifar10
  elif config.data.dataset == 'imagenet32':
    task = 'downsampled_imagenet/32x32'
    subset = 'validation'
    preprocess_fn = partial(_preprocess_cifar10, label_key=None)
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must be divisible by "
                     f"the number of devices ({jax.device_count()}).")
  per_device_batch_size = batch_size // jax.device_count()

  dataset_builder = tfds.builder(task)

  eval_split = deterministic_data.get_read_instruction_for_host(
      subset, dataset_builder.info.splits[subset].num_examples)
  batch_dims = [jax.local_device_count(), per_device_batch_size]

  test_ds = deterministic_data.create_dataset(
      dataset_builder,
      split=eval_split,
      num_epochs=num_epochs,
      shuffle=False,
      batch_dims=batch_dims,
      preprocess_fn=preprocess_fn,
      prefetch_size=tf.data.experimental.AUTOTUNE)

  return test_ds


def _preprocess_cifar10_augment_with_channel_flip(features):
  img = features['image']
  img = tf.cast(img, 'float32')

  # random left/right flip
  _img = tf.image.flip_left_right(img)
  aug = tf.random.uniform(shape=[]) > 0.5
  img = tf.where(aug, _img, img)

  # random 90 degree rotations
  u = tf.random.uniform(shape=[])
  k = tf.cast(tf.math.ceil(3. * u), tf.int32)
  _img = tf.image.rot90(img, k=k)
  _aug = tf.random.uniform(shape=[]) > 0.5
  img = tf.where(_aug, _img, img)
  aug = aug | _aug

  # random color channel flips
  _img = tf.transpose(img, [2, 0, 1])
  _img = tf.random.shuffle(_img)
  _img = tf.transpose(_img, [1, 2, 0])
  _aug = tf.random.uniform(shape=[]) > 0.5
  img = tf.where(_aug, _img, img)
  aug = aug | _aug

  aug = tf.cast(aug, 'uint8')

  return {'images': img, 'conditioning': aug}