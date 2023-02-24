# Copyright 2023 The TensorFlow GNN Authors. All Rights Reserved.
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
# ==============================================================================
"""Metrics for unsupervised embedding evaluation."""
from __future__ import annotations

import tensorflow as tf


@tf.function
def self_clustering(
    representations: tf.Tensor, *, subtract_mean: bool = False
) -> tf.Tensor:
  """Self-clustering metric implementation.

  Computes a metric that measures how well distributed representations are, if
  projected on the unit sphere. If `subtract_mean` is True, we additionally
  remove the mean from representations. The metric has a range of
  (-0.5, 1]. It achieves its maximum of 1 if representations collapse to a
  single point, and it is approximately 0 if representations are distributed
  randomly on the sphere. In theory, it can achieve negative values if the
  points are maximally equiangular, although this is very rare in practice.

  Args:
    representations: Input representations.
    subtract_mean: Whether to subtract the mean from representations.

  Returns:
    Metric value as scalar `tf.Tensor`.
  """
  if representations.shape.rank != 2:
    raise ValueError(f"Expected 2D tensor (got shape {representations.shape})")
  if subtract_mean:
    representations -= tf.reduce_mean(representations, axis=0)
  representations, _ = tf.linalg.normalize(representations, ord=2, axis=1)
  batch_size, feature_dim = tf.unstack(
      tf.cast(tf.shape(representations), tf.float32)
  )
  if tf.math.reduce_all(tf.math.is_nan(representations)):
    # If representations are completely collapsed, return 1.
    return tf.constant(1.0)
  expected = batch_size + batch_size * (batch_size - 1) / feature_dim
  actual = tf.reduce_sum(
      tf.square(tf.matmul(representations, representations, transpose_b=True))
  )
  return (actual - expected) / (batch_size * batch_size - expected)
