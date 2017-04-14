#!/usr/bin/env python

from model import Model
import argparse
import help
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=str, default="500,10")
parser.add_argument("--hidden_costs_scale", type=float, default=0.5)


class HiddenCostsModel(Model):
    def __init__(self, unparsed_args, _input, _truth_output):
        super().__init__(_input, _truth_output)

        args = parser.parse_args(unparsed_args)
        layers = help.get_layers_from_args(args)
        hidden_costs_scale = args.hidden_costs_scale

        current_input = self.input

        hidden_costs = None

        for i, hidden_layer_size in enumerate(layers):
            with tf.name_scope("layer" + str(i)):
                logits = help.logit_component(current_input, hidden_layer_size)

                layer_output = tf.nn.relu(logits)

                # Get cost as if the network ends here
                hidden_costs_output = tf.nn.relu(help.logit_component(layer_output, 10))
                logits_cost = self.create_cost_component(hidden_costs_output, "hidden_cost" + str(i))

                # Concat to `hidden_costs`
                if hidden_costs is None:
                    hidden_costs = logits_cost
                else:
                    hidden_costs = hidden_costs + logits_cost

                current_input = layer_output

        # Average `hidden_costs`
        hidden_costs = hidden_costs / len(layers)

        # Calculate output and final cost
        self.output = help.logit_component(current_input, 10)
        output_cost = self.create_cost_component(self.output, "output_cost")

        # Set cost to combination of final output cost and hidden costs
        self.cost = output_cost + (hidden_costs * hidden_costs_scale)

    def create_cost_component(self, logits, name):
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.truth_output, logits=logits), name=name)
        tf.summary.scalar(name + "_summary", cost)

        return cost
