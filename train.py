from __future__ import absolute_import
import tensorflow as tf
from configs import configs
from squeezenext_model import Model
import argparse
import numpy as np
import logging

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_dir', type=str, required=True,
                    help='Location of model_dir')
parser.add_argument('--configuration', type=str, default="v_1_0_SqNxt_23",
                    help='Name of model config file')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size during training')
parser.add_argument('--num_examples_per_epoch', type=int, default=1281160,
                    help='Number of examples in one epoch')
parser.add_argument('--num_eval_examples', type=int, default=50000,
                    help='Number of examples in one eval epoch')
parser.add_argument('--num_epochs', type=int, default=120,
                    help='Number of epochs for training')
parser.add_argument('--training_file_pattern', type=str, required=True,
                    help='Glob for training tf records')
parser.add_argument('--validation_file_pattern', type=str, required=True,
                    help='Glob for validation tf records')
parser.add_argument('--eval_after_training', type=bool, default=True,
                    help='Run one eval after the '
                         'training finishes.')
parser.add_argument('--eval_every_n_epochs', type=int, default=2,
                    help='Run eval every N epochs')
parser.add_argument('--output_train_images', type=bool, default=True,
                    help='Whether to save image summary during training (Warning: can lead to large event file sizes).')
args = parser.parse_args()


def main(argv):
    del argv #not used

    steps_per_epoch = (args.num_examples_per_epoch/args.batch_size)
    config = configs[args.configuration]
    config["model_dir"] = args.model_dir
    config["output_train_images"] = args.output_train_images
    config["total_steps"] = args.num_epochs*steps_per_epoch

    model = Model(config,args.batch_size)

    classifier = tf.estimator.Estimator(
        model_dir=args.model_dir,
        model_fn=model.model_fn,
        params=config)
    tf.logging.info("Total steps = {}, num_epochs = {}, batch size = {}".format(config["total_steps"],args.num_epochs,args.batch_size))
    classifier.train(
        input_fn=lambda: model.input_fn(args.training_file_pattern),
        steps=1)
    last_step = 1
    for epochs in np.linspace(args.eval_every_n_epochs, args.num_epochs,num=args.num_epochs/args.eval_every_n_epochs, endpoint=True):
        classifier.evaluate(
            input_fn=lambda: model.input_fn(args.validation_file_pattern),
            steps=args.num_eval_examples/args.batch_size)
        train_steps = int(epochs)*steps_per_epoch
        tf.logging.info(
            "Running training from step = {} till step = {}".format(last_step,train_steps))

        last_step = train_steps
        classifier.train(
            input_fn=lambda: model.input_fn(args.training_file_pattern),
            steps=train_steps)
    if args.eval_after_training:
        classifier.evaluate(
            input_fn=lambda: model.input_fn(args.validation_file_pattern),
            steps=args.num_eval_examples/args.batch_size)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)