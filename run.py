#!/usr/bin/env python

from datetime import datetime
from simple_model import SimpleModel
from hidden_costs_model import HiddenCostsModel
from locally_connected_model import LocallyConnectedModel
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, default="/tmp/tensorflow/mnist/input_data", help="Directory for storing input data")
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--training_steps", type=int, default=10000)
parser.add_argument("--display_step", type=int, default=1000)
parser.add_argument("--hidden_costs_model", dest="model", action="store_const", const="hidden_costs_model")
parser.add_argument("--locally_connected_model", dest="model", action="store_const", const="locally_connected_model")


def main():
    args, unknown_args = parser.parse_known_args()
    learning_rate = args.learning_rate
    training_steps = args.training_steps
    display_step = args.display_step

    # Import data
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

    # Define input and model_truth_output output
    model_input = tf.placeholder(tf.float32, [None, 784], name="input")
    model_truth_output = tf.placeholder(tf.float32, [None, 10], name="model_truth_output")

    # Get model
    with tf.name_scope("model"):
        if args.model == "hidden_costs_model":
            model = HiddenCostsModel(unknown_args, model_input, model_truth_output)
        elif args.model == "locally_connected_model":
            model = LocallyConnectedModel(unknown_args, model_input, model_truth_output)
        else:
            model = SimpleModel(unknown_args, model_input, model_truth_output)

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.cost)

    # Define accuracy
    with tf.name_scope("accuracy_calculation"):
        correct_prediction = tf.equal(
            tf.argmax(model.output, 1), tf.argmax(model_truth_output, 1), name="correct_prediction")
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        tf.summary.scalar("summary", accuracy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Set up writers
    time_formatted = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = tf.summary.FileWriter("/tmp/nn_mnist_experiments/" + time_formatted + "/train", sess.graph)
    test_writer = tf.summary.FileWriter("/tmp/nn_mnist_experiments/" + time_formatted + "/test", sess.graph)
    all_summaries = tf.summary.merge_all()

    # Train
    for step_number in range(training_steps):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, all_train_summaries = sess.run(
            [optimizer, all_summaries],
            feed_dict={
                model_input: batch_xs,
                model_truth_output: batch_ys
            })

        train_writer.add_summary(all_train_summaries, step_number)

        if step_number % display_step == 0:
            # Test trained model
            all_test_summaries = sess.run(
                all_summaries,
                feed_dict={
                    model_input: mnist.test.images,
                    model_truth_output: mnist.test.labels
                })

            test_writer.add_summary(all_test_summaries, step_number)


if __name__ == "__main__":
    main()
