from __future__ import absolute_import

import tensorflow as tf
import multiprocessing

def center_crop(image_encoded,image_size,training,resize_size=256):
    image = tf.image.decode_jpeg(image_encoded,channels=3)
    image = tf.image.resize_images(image, [resize_size, resize_size])
    image = tf.reshape(image, [resize_size, resize_size,3])
    if training:
        image = tf.random_crop(image,[image_size,image_size,3])
        image = tf.image.random_flip_left_right(image)
    else:
        crop_min = tf.abs(resize_size / 2 - (image_size / 2))
        crop_max = crop_min+image_size
        image = image[crop_min:crop_max,crop_min:crop_max,:]
    return image

def minimum_crop(image_encoded,image_size):
    image_shape = tf.image.extract_jpeg_shape(image_encoded)
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size = tf.minimum(image_height,image_width)
    crop_ymin = tf.abs(image_height / 2 - (crop_size / 2))
    crop_xmin = tf.abs(image_width / 2 - (crop_size / 2))

    crop_window = tf.stack([crop_ymin, crop_xmin, crop_size, crop_size])
    image = tf.image.decode_and_crop_jpeg(
        image_encoded,
        crop_window,
    )

    return tf.image.resize_images(image, [image_size, image_size])
def _parse_function(example_proto, image_size, num_classes,training,mean_value=(123,117,104),method="crop"):
    schema = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
    }

    image_size = tf.cast(image_size,tf.int32)
    mean_value = tf.cast(tf.stack(mean_value),tf.float32)
    parsed_features = tf.parse_single_example(example_proto, schema)
    jpeg_image = parsed_features["image/encoded"]
    if method == "crop":
        image = center_crop(jpeg_image,image_size,training)
    elif method == "minimum":
        image = minimum_crop(jpeg_image,image_size)
    elif method == "resize":
        image = tf.image.decode_jpeg(jpeg_image)
        image = tf.image.resize_images(image, [image_size, image_size])
    else:
        raise("unknown image process method")
    image = image - mean_value
    label_idx = tf.cast(parsed_features['image/class/label'], dtype=tf.int32)-1
    label_vec = tf.one_hot(label_idx, num_classes)

    return {"image": tf.reshape(image,[image_size,image_size,3])}, {"class_idx": label_idx, "class_vec": label_vec}


class ReadTFRecords(object):
    def __init__(self, image_size, batch_size, num_classes):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes

    def __call__(self, glob_pattern,training=True):
        threads = multiprocessing.cpu_count()
        with tf.name_scope("tf_record_reader"):
            files = tf.data.Dataset.list_files(glob_pattern)
            dataset = files.apply(tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset, cycle_length=threads * 2))
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(16*self.batch_size))
            dataset = dataset.map(map_func=lambda example: _parse_function(example, self.image_size, self.num_classes,training=training),
                                  num_parallel_calls=threads * 2)
            dataset = dataset.batch(batch_size=self.batch_size)
            dataset = dataset.prefetch(buffer_size=16)
            return dataset.make_one_shot_iterator()
