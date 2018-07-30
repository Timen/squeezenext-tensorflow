from __future__ import absolute_import
import tensorflow as tf
from configs import configs
from squeezenext_model import Model
import argparse
import numpy as np
import tools
tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description='Training parser')
parser.add_argument('--model_dir', type=str, required=True,
                    help='Location of model_dir')
parser.add_argument('--configuration', type=str, default="v_1_0_SqNxt_23",
                    help='Name of model config file')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size during training')
parser.add_argument('--num_examples_per_epoch', type=int, default=1281167,
                    help='Number of examples in one epoch')
parser.add_argument('--num_eval_examples', type=int, default=50000,
                    help='Number of examples in one eval epoch')
parser.add_argument('--num_epochs', type=int, default=120,
                    help='Number of epochs for training')
parser.add_argument('--training_file_pattern', type=str, required=True,
                    help='Glob for training tf records')
parser.add_argument('--validation_file_pattern', type=str, required=True,
                    help='Glob for validation tf records')
parser.add_argument('--eval_every_n_secs', type=int, default=1800,
                    help='Run eval every N seconds, default is every half hour')
parser.add_argument('--output_train_images', type=bool, default=True,
                    help='Whether to save image summary during training (Warning: can lead to large event file sizes).')
parser.add_argument('--fine_tune_ckpt', type=str, default=None,
                    help='Ckpt used for initializing the variables')
args = parser.parse_args()


def main(argv):
    """
    Main function to start training
    :param argv:
        not used
    :return:
        None
    """
    del argv  # not used

    # calculate steps per epoch
    steps_per_epoch = (args.num_examples_per_epoch / args.batch_size)

    # setup config dictionary
    config = configs[args.configuration]
    config["model_dir"] = args.model_dir
    config["output_train_images"] = args.output_train_images
    config["total_steps"] = args.num_epochs * steps_per_epoch
    config["model_dir"] = args.model_dir
    config["fine_tune_ckpt"] = args.fine_tune_ckpt
    # init model class
    model = Model(config, args.batch_size)

    # create classifier
    classifier = tf.estimator.Estimator(
        model_dir=args.model_dir,
        model_fn=model.model_fn,
        params=config)
    tf.logging.info("Total steps = {}, num_epochs = {}, batch size = {}".format(config["total_steps"], args.num_epochs,
                                                                                args.batch_size))

    # setup train spec
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: model.input_fn(args.training_file_pattern, True),
                                        max_steps=config["total_steps"])

    # setup eval spec evaluating ever n seconds
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: model.input_fn(args.validation_file_pattern, False),
        steps=args.num_eval_examples / args.batch_size,
        throttle_secs=args.eval_every_n_secs)

    # run train and evaluate
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    classifier.evaluate(input_fn=lambda: model.input_fn(args.validation_file_pattern, False),
        steps=args.num_eval_examples / args.batch_size)




if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
