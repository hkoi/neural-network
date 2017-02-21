import os

import tensorflow as tf

IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
NUM_CLASSES = len(CLASSES)
FEATURES = {
    'image': tf.FixedLenFeature([], dtype=tf.string),
    'label': tf.FixedLenFeature([1], dtype=tf.int64)
}
NAME = "mnist"

def training_examples(batch_size):
    reader = tf.TFRecordReader()
    queue = tf.train.string_input_producer([os.path.join("data", "mnist_training")])
    _, serialized_example = reader.read_up_to(queue, num_records=100000)
    features = tf.parse_example(serialized_example, features=FEATURES)
    image_feature = tf.cast(tf.decode_raw(features['image'], tf.uint8), dtype=tf.float32)
    image_feature = tf.reshape(image_feature, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    image_feature = tf.map_fn(lambda img:tf.image.per_image_standardization(img), image_feature)
    label_feature = features['label']
    image, label = tf.train.shuffle_batch(
        [image_feature, label_feature],
        enqueue_many = True,
        batch_size = batch_size,
        capacity = 50000,
        min_after_dequeue = 20000
    )
    return image, tf.reshape(label, [-1])

def validation_examples():
    reader = tf.TFRecordReader()
    queue = tf.train.string_input_producer([os.path.join("data", "mnist_validation")])
    _, serialized_example = reader.read_up_to(queue, num_records=100000)
    features = tf.parse_example(serialized_example, features=FEATURES)
    image = tf.cast(tf.decode_raw(features['image'], tf.uint8), dtype=tf.float32)
    image = tf.reshape(image, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    image = tf.map_fn(lambda img:tf.image.per_image_standardization(img), image)
    label = features['label']
    return image, tf.reshape(label, [-1])

def single_example(path):
    image_data = tf.read_file(path)
    image = tf.image.decode_png(image_data, channels=1)
    image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.image.per_image_standardization(image)
    return image
