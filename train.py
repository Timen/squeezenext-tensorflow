import tensorflow as tf
from configs import configs
from absl import flags
from squeezenext_model import Model

flags.DEFINE_string('model_dir', None, 'Location of model_dir')
flags.DEFINE_string('configuration', "v_1_0_SqNxt_23", 'Location of model_dir')
flags.DEFINE_integer('train_batch_size', 64, 'training batch size')
flags.DEFINE_integer('eval_steps', 5000, 'evaluation steps')

flags.DEFINE_string(
    'training_file_pattern', None,
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string(
    'validation_file_pattern', None,
    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
flags.DEFINE_integer('num_examples_per_epoch', 120000,
                     'Number of examples in one epoch')
flags.DEFINE_integer('num_epochs', 15, 'Number of epochs for training')
flags.DEFINE_string('mode', 'train',
                    'Mode to run: train or eval (default: train)')
flags.DEFINE_bool('eval_after_training', False, 'Run one eval after the '
                  'training finishes.')
FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.

    assert FLAGS.configuration in configs, "configuration not found"
    config = configs[FLAGS.configuration]
    model = Model(config)
    classifier = tf.estimator.Estimator(
        model_fn=model.model_fn,
        params=config)
    classifier.train(
        input_fn=lambda: model.input_fn("",FLAGS.train_batch_size),
        steps=1000)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)