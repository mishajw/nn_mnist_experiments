#!/usr/bin/env python

from model import Model
import argparse
import help
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--locally_connected_width", type=int, default=5)
parser.add_argument("--locally_connected_height", type=int, default=5)
parser.add_argument("--locally_connected_iterations", type=int, default=10)


class LocallyConnectedModel(Model):
    def __init__(self, unparsed_args, _input, _truth_output):
        super().__init__(_input, _truth_output)

        args = parser.parse_args(unparsed_args)

        self.create_guess_component(
            args.locally_connected_width, args.locally_connected_height, args.locally_connected_iterations)

        with tf.name_scope("cost"):
            self.create_cost_component()

    def create_guess_component(self, width, height, iterations):
        activations = tf.Variable(tf.zeros([width, height]), trainable=False, name="activations")
        horizontal_weights = tf.Variable(tf.zeros([width - 1, height]), name="horizontal_weights")
        vertical_weights = tf.Variable(tf.zeros([width, height - 1]), name="vertical_weights")

        for _ in range(iterations):
            from_left = tf.concat([horizontal_weights, tf.zeros([1, height])], 0) * activations
            from_right = tf.concat([tf.zeros([1, height]), horizontal_weights], 0) * activations
            from_down = tf.concat([vertical_weights, tf.zeros([width, 1])], 1) * activations
            from_up = tf.concat([tf.zeros([width, 1]), vertical_weights], 1) * activations

            activations = \
                tf.concat([tf.zeros([1, height]), tf.slice(from_left, [0, 0], [width - 1, height])], 0) + \
                tf.concat([tf.slice(from_right, [1, 0], [width - 1, height]), tf.zeros([1, height])], 0) + \
                tf.concat([tf.zeros([width, 1]), tf.slice(from_down, [0, 0], [width, height - 1])], 1) + \
                tf.concat([tf.slice(from_up, [0, 1], [width, height - 1]), tf.zeros([width, 1])], 1)

        self.output = help.logit_component(self.input, 10)

    def create_cost_component(self):
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.truth_output, logits=self.output), name="cost")
        tf.summary.scalar("summary", self.cost)
