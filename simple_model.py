#!/usr/bin/env python

from model import Model
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=str, default="500,10")


class SimpleModel(Model):
    def __init__(self, unparsed_args, _input, _truth_output):
        super().__init__(_input, _truth_output)

        args = parser.parse_args(unparsed_args)
        layers = [int(s) for s in args.layers.split(",")]

        self.create_guess_component(layers)

        self.create_cost_component()

    def create_guess_component(self, layers):
        current_input = self.input

        for i, hidden_layer_size in enumerate(layers):
            with tf.name_scope("layer" + str(i)):
                logits = self.logit_component(current_input, hidden_layer_size)
                current_input = tf.nn.relu(logits)

        self.output = self.logit_component(current_input, 10)

    def create_cost_component(self):
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.truth_output, logits=self.output), name="cost")
        tf.summary.scalar("cost_summary", self.cost)

    @staticmethod
    def tensor_summary(t):
        t_mean = tf.reduce_mean(t)
        t_stddev = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(t, t_mean))))

        tf.summary.scalar("mean", t_mean)
        tf.summary.scalar("stddev", t_stddev)
        tf.summary.scalar("max", tf.reduce_max(t))
        tf.summary.scalar("min", tf.reduce_min(t))

    def logit_component(self, layer_input, size):
        weights = tf.Variable(
            tf.random_normal([int(layer_input.shape[1]), size]),
            name="weights")

        biases = tf.Variable(
            tf.zeros([size]),
            name="biases")

        logits = tf.matmul(layer_input, weights) + biases

        with tf.name_scope("summary"):
            with tf.name_scope("weights"):
                self.tensor_summary(weights)

            with tf.name_scope("biases"):
                self.tensor_summary(biases)

            negative_logits = tf.reduce_mean(tf.cast(tf.equal(tf.sign(logits), -1), tf.float32))
            tf.summary.scalar("negative_logits", negative_logits)

        return logits
