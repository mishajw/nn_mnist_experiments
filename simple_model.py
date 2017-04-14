#!/usr/bin/env python

import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=str, default="500,10")


class SimpleModel:
    def __init__(self, unparsed_args):
        args = parser.parse_args(unparsed_args)
        layers = [int(s) for s in args.layers.split(",")]

        self.image_input = tf.placeholder(tf.float32, [None, 784])

        current_input = self.image_input

        for i, hidden_layer_size in enumerate(layers):
            weights = tf.Variable(
                tf.zeros([int(current_input.shape[1]), hidden_layer_size]),
                "weights" + str(i))

            biases = tf.Variable(
                tf.zeros([hidden_layer_size]),
                "biases" + str(i))

            current_input = tf.nn.sigmoid(tf.matmul(current_input, weights) + biases)

        self.guesses = tf.nn.softmax(current_input)
