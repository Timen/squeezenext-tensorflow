import tensorflow as tf
from configs import configs
from absl import flags
from squeezenext_model import Model

flags.DEFINE_string('configuration', "v_1_0_SqNxt_23", 'Location of model_dir')
flags.DEFINE_integer('train_batch_size', 64, 'training batch size')

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.

    assert FLAGS.configuration in configs, "configuration not found"
    config = configs[FLAGS.configuration]
    model = Model(config)
    input = model.input_fn("",FLAGS.train_batch_size)
    model_net = model.model_fn(input[0],input[1],tf.estimator.ModeKeys.TRAIN,config)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
