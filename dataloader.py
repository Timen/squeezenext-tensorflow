import tensorflow as tf
import multiprocessing


def _parse_function(example_proto, image_size, num_classes,mean_value):
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

    parsed_features = tf.parse_single_example(example_proto, schema)
    jpeg_image = parsed_features["image/encoded"]

    crop_height = (image_size - tf.abs(parsed_features["image/height"] - image_size))
    crop_width = (image_size - tf.abs(parsed_features["image/width"] - image_size))
    crop_ymin = parsed_features["image/height"] / 2 - (crop_height / 2)
    crop_xmin = parsed_features["image/width"] / 2 - (crop_width / 2)

    crop_window = tf.cast(tf.concat([crop_ymin, crop_xmin, crop_height, crop_width],axis=0),tf.int32)
    image = tf.image.decode_and_crop_jpeg(
        jpeg_image,
        crop_window,
    )
    image = tf.cond(tf.logical_or(crop_height[0] < image_size, crop_width[0] < image_size),
                    lambda: tf.image.resize_images(image, [image_size, image_size]), lambda: tf.cast(image,tf.float32))
    image = image - mean_value
    label_idx = tf.cast(parsed_features['image/class/label'], dtype=tf.int32)
    label_vec = tf.one_hot(label_idx, num_classes)

    return {"image": tf.reshape(image,[image_size,image_size,3])}, {"class_idx": label_idx, "class_vec": label_vec}


class ReadTFRecords(object):
    def __init__(self, image_size, batch_size, num_classes,mean_value):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.mean_value = tf.cast(tf.stack(mean_value),tf.float32)

    def __call__(self, glob_pattern, shuffle_buffer_size=4096):
        threads = multiprocessing.cpu_count()

        files = tf.data.Dataset.list_files(glob_pattern)
        dataset = files.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=threads * 2))
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.map(map_func=lambda example: _parse_function(example, self.image_size, self.num_classes,self.mean_value),
                              num_parallel_calls=threads * 2)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(buffer_size=16)
        return dataset.make_one_shot_iterator()
