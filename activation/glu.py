# -*- coding: utf-8 -*-
import tensorflow as tf


def glu(act, n_units):
    """Generalized linear unit nonlinear activation."""
    return act[:, :n_units] * tf.nn.sigmoid(act[:, n_units:])
