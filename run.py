#!/usr/bin/env python

from datetime import datetime
from simple_model import SimpleModel
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, default="/tmp/tensorflow/mnist/input_data", help="Directory for storing input data")
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--training_steps", type=int, default=10000)
parser.add_argument("--display_step", type=int, default=1000)


def main():
    args, unknown_args = parser.parse_known_args()
    learning_rate = args.learning_rate
    training_steps = args.training_steps
    display_step = args.display_step

    # Import data
    mnist = input_data.read_data_sets(args.data_dir, one_hot=True)

    # Get model
    model = SimpleModel(unknown_args)

    # Define loss and optimizer
    truth = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=truth, logits=model.guesses))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    tf.summary.scalar("cross_entropy", cross_entropy)

    # Define accuracy
    correct_prediction = tf.equal(tf.argmax(model.guesses, 1), tf.argmax(truth, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

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
            [train_step, all_summaries],
            feed_dict={
                model.image_input: batch_xs,
                truth: batch_ys
            })

        train_writer.add_summary(all_train_summaries, step_number)

        if step_number % display_step == 0:
            # Test trained model
            all_test_summaries = sess.run(
                all_summaries,
                feed_dict={
                    model.image_input: mnist.test.images,
                    truth: mnist.test.labels
                })

            test_writer.add_summary(all_test_summaries, step_number)


if __name__ == "__main__":
    main()
