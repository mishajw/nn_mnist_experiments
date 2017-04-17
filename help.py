#!/usr/bin/env python

import tensorflow as tf


def tensor_summary(t):
    t_mean = tf.reduce_mean(t)
    t_stddev = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(t, t_mean))))

    tf.summary.scalar("mean", t_mean)
    tf.summary.scalar("stddev", t_stddev)
    tf.summary.scalar("max", tf.reduce_max(t))
    tf.summary.scalar("min", tf.reduce_min(t))


def logit_component(layer_input, size):
    weights = tf.Variable(
        tf.random_normal([int(layer_input.shape[1]), size]),
        name="weights")

    biases = tf.Variable(
        tf.zeros([size]),
        name="biases")

    logits = tf.matmul(layer_input, weights) + biases

    with tf.name_scope("summary"):
        with tf.name_scope("weights"):
            tensor_summary(weights)

        with tf.name_scope("biases"):
            tensor_summary(biases)

        negative_logits = tf.reduce_mean(tf.cast(tf.equal(tf.sign(logits), -1), tf.float32))
        tf.summary.scalar("negative_logits", negative_logits)

    return logits


def get_int_list_from_string(string):
    return [int(s) for s in string.split(",")]
