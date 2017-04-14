#!/usr/bin/env python

from model import Model
import argparse
import help
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=str, default="500,10")


class SimpleModel(Model):
    def __init__(self, unparsed_args, _input, _truth_output):
        super().__init__(_input, _truth_output)

        args = parser.parse_args(unparsed_args)
        layers = help.get_layers_from_args(args)

        self.create_guess_component(layers)

        with tf.name_scope("cost"):
            self.create_cost_component()

    def create_guess_component(self, layers):
        current_input = self.input

        for i, hidden_layer_size in enumerate(layers):
            with tf.name_scope("layer" + str(i)):
                logits = help.logit_component(current_input, hidden_layer_size)
                current_input = tf.nn.relu(logits)

        with tf.name_scope("final_layer"):
            self.output = help.logit_component(current_input, 10)

    def create_cost_component(self):
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.truth_output, logits=self.output), name="cost")
        tf.summary.scalar("summary", self.cost)


