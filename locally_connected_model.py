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
        with tf.name_scope("preprocess"):
            preprocess = help.logit_component(self.input, height)

        with tf.name_scope("activations_construction"):
            stack = tf.stack([tf.shape(self.input)[0], width - 1, height])
            unknown_activations = tf.fill(stack, 0.0)
            activations = tf.concat([tf.expand_dims(preprocess, 1), unknown_activations], 1)

        horizontal_weights = tf.Variable(tf.random_normal([width - 1, height]), name="horizontal_weights")
        vertical_weights = tf.Variable(tf.random_normal([width, height - 1]), name="vertical_weights")
        horizontal_biases = tf.Variable(tf.zeros([width - 1, height]), name="horizontal_biases")
        vertical_biases = tf.Variable(tf.zeros([width, height - 1]), name="vertical_biases")

        for i in range(iterations):
            with tf.name_scope("iteration" + str(i)):
                from_left = tf.concat([horizontal_weights, tf.zeros([1, height])], 0) * activations
                from_right = tf.concat([tf.zeros([1, height]), horizontal_weights], 0) * activations
                from_down = tf.concat([vertical_weights, tf.zeros([width, 1])], 1) * activations
                from_up = tf.concat([tf.zeros([width, 1]), vertical_weights], 1) * activations

                # TODO: Check why relu doesn't work here
                activations = tf.nn.sigmoid( \
                    tf.concat([
                        tf.zeros([tf.shape(self.input)[0], 1, height]),
                        tf.slice(from_left, [0, 0, 0], [-1, width - 1, height]) + horizontal_biases], 1) + \
                    tf.concat([
                        tf.slice(from_right, [0, 1, 0], [-1, width - 1, height]) + horizontal_biases,
                        tf.zeros([tf.shape(self.input)[0], 1, height])], 1) + \
                    tf.concat([
                        tf.zeros([tf.shape(self.input)[0], width, 1]),
                        tf.slice(from_down, [0, 0, 0], [-1, width, height - 1]) + vertical_biases], 2) + \
                    tf.concat([
                        tf.slice(from_up, [0, 0, 1], [-1, width, height - 1]) + vertical_biases,
                        tf.zeros([tf.shape(self.input)[0], width, 1])], 2))

        with tf.name_scope("postprocess"):
            locally_connected_output = tf.squeeze(tf.slice(activations, [0, width - 2, 0], [-1, 1, height]), [1])
            self.output = help.logit_component(locally_connected_output, 10)

        with tf.name_scope("activations_summary"):
            help.tensor_summary(activations)

        with tf.name_scope("horizontal_weights_summary"):
            help.tensor_summary(horizontal_weights)

        with tf.name_scope("vertical_weights_summary"):
            help.tensor_summary(vertical_weights)

        with tf.name_scope("horizontal_biases_summary"):
            help.tensor_summary(horizontal_biases)

        with tf.name_scope("vertical_biases_summary"):
            help.tensor_summary(vertical_biases)

    def create_cost_component(self):
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.truth_output, logits=self.output), name="cost")
        tf.summary.scalar("summary", self.cost)
