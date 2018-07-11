import tensorflow as tf
from configs import configs
from squeezenext_model import Model
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_dir', type=str, required=True,
                    help='Location of model_dir')
parser.add_argument('--configuration', type=str, default="v_1_0_SqNxt_23",
                    help='Name of model config file')
parser.add_argument('--train_batch_size', type=int, default=64,
                    help='Batch size during training')
parser.add_argument('--num_examples_per_epoch', type=int, default=1300000,
                    help='Number of examples in one epoch')
parser.add_argument('--num_epochs', type=int, default=120,
                    help='Number of epochs for training')
parser.add_argument('--training_file_pattern', type=str, required=True,
                    help='Glob for training tf records')
parser.add_argument('--validation_file_pattern', type=str, required=True,
                    help='Glob for validation tf records')
parser.add_argument('--eval_after_training', type=bool, default=True,
                    help='Run one eval after the '
                         'training finishes.')

args = parser.parse_args()


def main(argv):
    del argv #not used

    config = configs[args.configuration]
    config["model_dir"] = args.model_dir

    model = Model(config)

    classifier = tf.estimator.Estimator(
        model_dir=args.model_dir,
        model_fn=model.model_fn,
        params=config)
    classifier.train(
        input_fn=lambda: model.input_fn("",args.train_batch_size),
        steps=1000)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)