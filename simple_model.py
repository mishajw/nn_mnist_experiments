#!/usr/bin/env python

import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=str, default="500,10")


class SimpleModel:
    def __init__(self, unparsed_args):
        args = parser.parse_args(unparsed_args)
        layers = [int(s) for s in args.layers.split(",")]

        self.input = tf.placeholder(tf.float32, [None, 784], name="input")

        current_input = self.input

        for i, hidden_layer_size in enumerate(layers):
            with tf.name_scope("layer" + str(i)):
                weights = tf.Variable(
                    tf.zeros([int(current_input.shape[1]), hidden_layer_size]),
                    name="weights")

                biases = tf.Variable(
                    tf.zeros([hidden_layer_size]),
                    name="biases")

                current_input = tf.nn.sigmoid(tf.matmul(current_input, weights) + biases)

        self.output = tf.nn.softmax(current_input, name="output")
