#!/usr/bin/env python

import tensorflow as tf


class SimpleModel:
    HIDDEN_LAYER_SIZES = [50, 10]

    def __init__(self):
        self.image_input = tf.placeholder(tf.float32, [None, 784])

        current_input = self.image_input

        for i, hidden_layer_size in enumerate(self.HIDDEN_LAYER_SIZES):
            weights = tf.Variable(
                tf.zeros([int(current_input.shape[1]), hidden_layer_size]),
                "weights" + str(i))

            biases = tf.Variable(
                tf.zeros([hidden_layer_size]),
                "biases" + str(i))

            current_input = tf.nn.sigmoid(tf.matmul(current_input, weights) + biases)

        self.guesses = tf.nn.softmax(current_input)
