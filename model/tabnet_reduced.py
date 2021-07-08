# -*- coding: utf-8 -*-
"""Modified reduced TabNet model."""
import numpy as np
import sys

sys.path.append("..")

import tensorflow.compat.v1 as tf

from activation.entmax import entmax15
from activation.glu import glu


class TabNetReduced(object):
    """Reduced TabNet model class."""

    def __init__(
        self,
        columns,
        num_features,
        feature_dim,
        output_dim,
        num_decision_steps,
        relaxation_factor,
        batch_momentum,
        virtual_batch_size,
        num_classes,
        epsilon=0.00001,
    ):
        """Initializes a reduced TabNet instance.

    Args:
      columns: The Tensorflow column names for the dataset.
      num_features: The number of input features (i.e the number of columns for
        tabular data assuming each feature is represented with 1 dimension).
      feature_dim: Dimensionality of the hidden representation in feature
        transformation block. Each layer first maps the representation to a
        2*feature_dim-dimensional output and half of it is used to determine the
        nonlinearity of the GLU activation where the other half is used as an
        input to GLU, and eventually feature_dim-dimensional output is
        transferred to the next layer.
      output_dim: Dimensionality of the outputs of each decision step, which is
        later mapped to the final classification or regression output.
      num_decision_steps: Number of sequential decision steps.
      relaxation_factor: Relaxation factor that promotes the reuse of each
        feature at different decision steps. When it is 1, a feature is enforced
        to be used only at one decision step and as it increases, more
        flexibility is provided to use a feature at multiple decision steps.
      batch_momentum: Momentum in ghost batch normalization.
      virtual_batch_size: Virtual batch size in ghost batch normalization. The
        overall batch size should be an integer multiple of virtual_batch_size.
      num_classes: Number of output classes.
      epsilon: A small number for numerical stability of the entropy calcations.

    Returns:
      A reduced TabNet instance.
    """

        self.columns = columns
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_classes = num_classes
        self.epsilon = epsilon

    def encoder(self, data, reuse, is_training):
        """Reduced TabNet encoder model."""

        with tf.variable_scope("Encoder", reuse=reuse):

            # Reads and normalizes input features.
            features = tf.feature_column.input_layer(data, self.columns)
            features = tf.layers.batch_normalization(
                features, training=is_training, momentum=self.batch_momentum
            )
            batch_size = tf.shape(features)[0]

            # Initializes decision-step dependent variables.
            output_aggregated = tf.zeros([batch_size, self.output_dim])
            masked_features = features
            mask_values = tf.zeros([batch_size, self.num_features])
            aggregated_mask_values = tf.zeros([batch_size, self.num_features])
            complementary_aggregated_mask_values = tf.ones([batch_size, self.num_features])
            total_entropy = 0

            v_b = self.virtual_batch_size if is_training else 1
            # Feature transformer: a sort of recurrent structure
            # TODO: can we automate number of decision steps needed?
            for ni in range(self.num_decision_steps):
                # Feature transformer with one shared and one decision step dependent
                # blocks is used below. This departs from the original model
                reuse_flag = ni > 0

                # shared because of the same name
                transform_f1 = tf.layers.dense(
                    masked_features,
                    self.feature_dim * 2,
                    name="Transform_f1",
                    reuse=reuse_flag,
                    use_bias=False,
                )
                transform_f1 = tf.layers.batch_normalization(
                    transform_f1,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b,
                )
                transform_f1 = glu(transform_f1, self.feature_dim)

                # step dependent
                transform_f2 = tf.layers.dense(
                    transform_f1,
                    self.feature_dim * 2,
                    name="Transform_f1" + str(ni),
                    use_bias=False,
                )
                transform_f2 = tf.layers.batch_normalization(
                    transform_f2,
                    training=is_training,
                    momentum=self.batch_momentum,
                    virtual_batch_size=v_b,
                )
                transform_f2 = (glu(transform_f2, self.feature_dim) + transform_f1) * np.sqrt(0.5)

                if ni > 0:
                    decision_out = tf.nn.relu(transform_f2[:, : self.output_dim])

                    # Decision aggregation.
                    output_aggregated += decision_out

                    # Aggregated masks are used for visualization of the
                    # feature importance attributes.
                    scale_agg = tf.reduce_sum(decision_out, axis=1, keep_dims=True) / (
                        self.num_decision_steps - 1
                    )
                    aggregated_mask_values += mask_values * scale_agg

                features_for_coef = transform_f2[:, self.output_dim :]

                # Attentive transformer
                if ni < self.num_decision_steps - 1:

                    # Determines the feature masks via linear and nonlinear
                    # transformations, taking into account of aggregated feature use.
                    mask_values = tf.layers.dense(
                        features_for_coef,
                        self.num_features,
                        name="Transform_coef" + str(ni),
                        use_bias=False,
                    )
                    mask_values = tf.layers.batch_normalization(
                        mask_values,
                        training=is_training,
                        momentum=self.batch_momentum,
                        virtual_batch_size=v_b,
                    )
                    mask_values *= complementary_aggregated_mask_values
                    # replace sparsemax with entmax 1.5
                    mask_values = entmax15(mask_values)

                    # Relaxation factor controls the amount of reuse of features between
                    # different decision blocks and updated with the values of
                    # coefficients.
                    complementary_aggregated_mask_values *= self.relaxation_factor - mask_values

                    # Entropy is used to penalize the amount of sparsity in feature
                    # selection.
                    total_entropy += tf.reduce_mean(
                        tf.reduce_sum(-mask_values * tf.log(mask_values + self.epsilon), axis=1)
                    ) / (self.num_decision_steps - 1)

                    # Feature selection.
                    masked_features = tf.multiply(mask_values, features)

                    # Visualization of the feature selection mask at decision step ni
                    tf.summary.image(
                        "Mask_for_step" + str(ni),
                        tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
                        max_outputs=1,
                    )

            # Visualization of the aggregated feature importances
            tf.summary.image(
                "Aggregated_mask",
                tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),
                max_outputs=1,
            )

            return output_aggregated, total_entropy

    def classify(self, activations, reuse):
        """Reduced TabNet classify block."""

        with tf.variable_scope("Classify", reuse=reuse):
            logits = tf.layers.dense(activations, self.num_classes, use_bias=False)
            predictions = tf.nn.softmax(logits)
            return logits, predictions

    def regress(self, activations, reuse):
        """Reduced TabNet regress block."""

        with tf.variable_scope("Regress", reuse=reuse):
            predictions = tf.layers.dense(activations, 1)
            return predictions
