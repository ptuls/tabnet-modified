# -*- coding: utf-8 -*-
import functools
import tensorflow as tf
from tensorflow.python.framework import dtypes


def set_defaults(int_columns, bool_columns, float_columns, str_columns):
    return (
        [[0] for col in int_columns]
        + [[""] for col in bool_columns]
        + [[0.0] for col in float_columns]
        + [[""] for col in str_columns]
        + [[-1]]
    )


def get_columns(int_columns, encoded_categorical_columns, bool_columns, float_columns, str_columns):
    """Get the representations for all input columns."""

    columns = []
    if float_columns:
        columns += [
            tf.feature_column.numeric_column(ci, dtype=dtypes.float32) for ci in float_columns
        ]
    if int_columns:
        columns += [tf.feature_column.numeric_column(ci, dtype=dtypes.int32) for ci in int_columns]
    if encoded_categorical_columns:
        columns += [
            tf.feature_column.numeric_column(ci, dtype=dtypes.int32)
            for ci in encoded_categorical_columns
        ]
    if str_columns:
        # pylint: disable=g-complex-comprehension
        str_nuniquess = len(set(str_columns))
        columns += [
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_hash_bucket(
                    ci, hash_bucket_size=int(3 * num)
                ),
                dimension=1,
            )
            for ci, num in zip(str_columns, str_nuniquess)
        ]
    if bool_columns:
        # pylint: disable=g-complex-comprehension
        columns += [
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_hash_bucket(ci, hash_bucket_size=3),
                dimension=1,
            )
            for ci in bool_columns
        ]
    return columns


def parse_csv(int_columns, bool_columns, float_columns, str_columns, label_column, value_column):
    """Parses a CSV file based on the provided column types."""
    defaults = set_defaults(int_columns, bool_columns, float_columns, str_columns)
    all_columns = int_columns + bool_columns + float_columns + str_columns + [label_column]
    columns = tf.decode_csv(value_column, record_defaults=defaults)
    features = dict(zip(all_columns, columns))
    label = features.pop(label_column)
    classes = tf.cast(label, tf.int32) - 1
    return features, classes


def input_fn(
    data_file,
    int_columns,
    bool_columns,
    float_columns,
    str_columns,
    label_column,
    num_epochs,
    shuffle,
    batch_size,
    n_buffer=50,
    n_parallel=16,
):
    """Function to read the input file and return the dataset.

    Args:
        data_file: Name of the file.
        num_epochs: Number of epochs.
        shuffle: Whether to shuffle the data.
        batch_size: Batch size.
        n_buffer: Buffer size.
        n_parallel: Number of cores for multi-core processing option.

    Returns:
        The Tensorflow dataset.
    """

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=n_buffer)

    parse_csv_partial = functools.partial(
        parse_csv,
        int_columns,
        bool_columns,
        float_columns,
        str_columns,
        label_column,
    )

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(parse_csv_partial, num_parallel_calls=n_parallel)

    # Repeat after shuffling, to prevent separate epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    return dataset
