# -*- coding: utf-8 -*-
import tensorflow as tf


def create_tf_example(row, int_columns, float_columns, label_column):
    features = {}

    for feat_name in int_columns:
        features[feat_name] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[row[feat_name].astype(dtype=int)])
        )

    for feat_name in float_columns:
        features[feat_name] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[row[feat_name]])
        )

    features[label_column] = tf.train.Feature(float_list=tf.train.FloatList(value=[row[feat_name]]))

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example


def write_tfrecords(df, path):
    with tf.python_io.TFRecordWriter(path) as writer:
        for index, row in df.iterrows():
            tf_example = create_tf_example(row)
            writer.write(tf_example.SerializeToString())
