# -*- coding: utf-8 -*-
"""Train the TabNet or reduced TabNet model on various datasets."""
import os
from absl import app
import numpy as np
import tensorflow as tf

from datetime import datetime
from config.covertype import *
from model import tabnet, tabnet_reduced
from util import data_helper, logging

logger = logging.create_logger()

# Run Tensorflow on GPU 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def sort_col_names(feature_cols):
    column_names = sorted(feature_cols)
    logger.info(
        "Ordered column names, corresponding to the indexing in Tensorboard visualization")
    for fi in range(len(column_names)):
        logger.info(str(fi) + " : " + column_names[fi])


def main(unused_argv):
    # column order
    feature_columns = INT_COLUMNS + ENCODED_CATEGORICAL_COLUMNS + BOOL_COLUMNS + STR_COLUMNS + FLOAT_COLUMNS
    all_columns = feature_columns + [LABEL_COLUMN]

    # Fix random seeds
    tf.set_random_seed(SEED)
    np.random.seed(SEED)

    input_columns = data_helper.get_columns(
        INT_COLUMNS, BOOL_COLUMNS, FLOAT_COLUMNS, STR_COLUMNS)

    # Define the TabNet model
    tabnet_model = (
        (
            tabnet_reduced.TabNetReduced(
                columns=input_columns,
                num_features=NUM_FEATURES,
                feature_dim=FEATURE_DIM,
                output_dim=OUTPUT_DIM,
                num_decision_steps=NUM_DECISION_STEPS,
                relaxation_factor=RELAXATION_FACTOR,
                batch_momentum=BATCH_MOMENTUM,
                virtual_batch_size=VIRTUAL_BATCH_SIZE,
                num_classes=NUM_CLASSES,
            )
        )
        if REDUCED
        else (
            tabnet.TabNet(
                columns=input_columns,
                num_features=NUM_FEATURES,
                feature_dim=FEATURE_DIM,
                output_dim=OUTPUT_DIM,
                num_decision_steps=NUM_DECISION_STEPS,
                relaxation_factor=RELAXATION_FACTOR,
                batch_momentum=BATCH_MOMENTUM,
                virtual_batch_size=VIRTUAL_BATCH_SIZE,
                num_classes=NUM_CLASSES,
            )
        )
    )

    sort_col_names(feature_columns)

    # Input sampling
    train_batch = data_helper.input_fn(
        TRAIN_FILE, num_epochs=MAX_STEPS, shuffle=True, batch_size=BATCH_SIZE
    )
    test_batch = data_helper.input_fn(
        TEST_FILE, num_epochs=MAX_STEPS, shuffle=False, batch_size=data_helper.N_TEST_SAMPLES
    )

    train_iter = train_batch.make_initializable_iterator()
    test_iter = test_batch.make_initializable_iterator()

    feature_train_batch, label_train_batch = train_iter.get_next()
    feature_test_batch, label_test_batch = test_iter.get_next()

    # Define the model and losses
    encoded_train_batch, total_entropy = tabnet_model.encoder(
        feature_train_batch, reuse=False, is_training=True
    )

    logits_orig_batch, _ = tabnet_model.classify(
        encoded_train_batch, reuse=False)

    softmax_orig_key_op = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_orig_batch, labels=label_train_batch
        )
    )

    train_loss_op = softmax_orig_key_op + SPARSITY_LOSS_WEIGHT * total_entropy
    tf.summary.scalar("Total loss", train_loss_op)

    # Optimization step
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        INIT_LEARNING_RATE, global_step=global_step, decay_steps=DECAY_EVERY, decay_rate=DECAY_RATE
    )
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gvs = optimizer.compute_gradients(train_loss_op)
        capped_gvs = [
            (tf.clip_by_value(grad, -GRADIENT_THRESH, GRADIENT_THRESH), var) for grad, var in gvs
        ]
        train_op = optimizer.apply_gradients(
            capped_gvs, global_step=global_step)

    # Model evaluation
    # Test performance
    encoded_test_batch, _ = tabnet_model.encoder(
        feature_test_batch, reuse=True, is_training=False)

    _, prediction_test = tabnet_model.classify(encoded_test_batch, reuse=True)

    predicted_labels = tf.cast(tf.argmax(prediction_test, 1), dtype=tf.int32)
    test_eq_op = tf.equal(predicted_labels, label_test_batch)
    test_acc_op = tf.reduce_mean(tf.cast(test_eq_op, dtype=tf.float32))
    tf.summary.scalar("Test accuracy", test_acc_op)

    # Training setup
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = MODEL_NAME + f"_{current_time}"
    init = tf.initialize_all_variables()
    init_local = tf.local_variables_initializer()
    init_table = tf.tables_initializer(name="Initialize_all_tables")
    saver = tf.train.Saver()
    summaries = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(
            "./tflog/" + model_name, sess.graph)

        sess.run(init)
        sess.run(init_local)
        sess.run(init_table)
        sess.run(train_iter.initializer)
        sess.run(test_iter.initializer)

        for step in range(1, MAX_STEPS + 1):
            if step % DISPLAY_STEP == 0:
                _, train_loss, merged_summary = sess.run(
                    [train_op, train_loss_op, summaries])
                summary_writer.add_summary(merged_summary, step)
                logger.info("Step " + str(step) + ", Training Loss = " +
                      "{:.4f}".format(train_loss))
            else:
                _ = sess.run(train_op)

            if step % TEST_STEP == 0:
                feed_arr = [vars()["summaries"], vars()["test_acc_op"]]

                test_arr = sess.run(feed_arr)
                merged_summary = test_arr[0]
                test_acc = test_arr[1]

                logger.info("Step " + str(step) + ", Test Accuracy = " +
                      "{:.4f}".format(test_acc))
                summary_writer.add_summary(merged_summary, step)

            if step % SAVE_STEP == 0:
                saver.save(sess, "./checkpoints/" + model_name + ".ckpt")


if __name__ == "__main__":
    app.run(main)
