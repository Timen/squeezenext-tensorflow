from __future__ import absolute_import

import tensorflow as tf
import multiprocessing


def _parse_function(example_proto, image_size, num_classes,mean_value=[104,117,123]):
    schema = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        'image/height': tf.FixedLenFeature([1], dtype=tf.int64),
        'image/width': tf.FixedLenFeature([1], dtype=tf.int64),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }

    image_size = tf.cast(image_size,tf.int32)
    mean_value = tf.cast(tf.stack(mean_value),tf.float32)
    parsed_features = tf.parse_single_example(example_proto, schema)
    jpeg_image = parsed_features["image/encoded"]
    image_height = tf.cast(parsed_features["image/height"],tf.int32)[0]
    image_width =  tf.cast(parsed_features["image/width"],tf.int32)[0]
    crop_height = tf.clip_by_value(image_size,0,image_height)
    crop_width = tf.clip_by_value(image_size,0,image_width)
    crop_ymin = tf.abs(image_height / 2 - (crop_height / 2))
    crop_xmin = tf.abs(image_width / 2 - (crop_width / 2))

    crop_window = tf.stack([crop_ymin, crop_xmin, crop_height, crop_width])
    image = tf.image.decode_and_crop_jpeg(
        jpeg_image,
        crop_window,
    )
    image = tf.cond(tf.logical_or(crop_height < image_size, crop_width < image_size),
                    lambda: tf.image.resize_images(image, [image_size, image_size]), lambda: tf.cast(image,tf.float32))
    image = image - mean_value
    label_idx = tf.cast(parsed_features['image/class/label'], dtype=tf.int32)
    label_vec = tf.one_hot(label_idx, num_classes)

    return {"image": tf.reshape(image,[image_size,image_size,3])}, {"class_idx": label_idx, "class_vec": label_vec}


class ReadTFRecords(object):
    def __init__(self, image_size, batch_size, num_classes):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes

    def __call__(self, glob_pattern):
        threads = multiprocessing.cpu_count()
        with tf.name_scope("tf_record_reader"):
            files = tf.data.Dataset.list_files(glob_pattern)
            dataset = files.apply(tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=threads * 2))
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(16*self.batch_size))
            dataset = dataset.map(map_func=lambda example: _parse_function(example, self.image_size, self.num_classes),
                                  num_parallel_calls=threads * 2)
            dataset = dataset.batch(batch_size=self.batch_size)
            dataset = dataset.prefetch(buffer_size=16)
            return dataset.make_one_shot_iterator()
