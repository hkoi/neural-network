import os

import tensorflow as tf

INPUT_SIZE = 48
IMAGE_SIZE = 32
CENTER_OFFSET = (INPUT_SIZE - IMAGE_SIZE) // 2
IMAGE_CHANNELS = 3
CLASSES = ["1neoplastic", "2impressionist", "3expressionist", "4colorfield"]
NUM_CLASSES = len(CLASSES)
FEATURES = {
    'image': tf.FixedLenFeature([], dtype=tf.string),
    'label': tf.FixedLenFeature([1], dtype=tf.int64)
}
NAME = "artclass"

def training_examples(batch_size):
    reader = tf.TFRecordReader()
    queue = tf.train.string_input_producer([os.path.join("data", "%s_training" % NAME)])
    _, serialized_example = reader.read_up_to(queue, num_records=100000)
    features = tf.parse_example(serialized_example, features=FEATURES)
    image_feature = tf.decode_raw(features['image'], tf.float32)
    image_feature = tf.reshape(image_feature, [-1, INPUT_SIZE, INPUT_SIZE, IMAGE_CHANNELS])
    label_feature = features['label']
    image, label = tf.train.shuffle_batch(
        [image_feature, label_feature],
        enqueue_many=True,
        batch_size=batch_size,
        capacity=50000,
        min_after_dequeue=20000
    )
    image = tf.map_fn(lambda img:tf.random_crop(img, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS]), image)
    image = tf.image.random_brightness(image, max_delta=63.0)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.maximum(image, 0.0)
    image = tf.minimum(image, 255.0)
    image = tf.map_fn(lambda img:tf.image.random_flip_up_down(img), image)
    image = tf.map_fn(lambda img:tf.image.random_flip_left_right(img), image)
    image = tf.map_fn(lambda img:tf.image.per_image_standardization(img), image)
    return image, tf.reshape(label, [-1])

def validation_examples():
    reader = tf.TFRecordReader()
    queue = tf.train.string_input_producer([os.path.join("data", "%s_validation" % NAME)])
    _, serialized_example = reader.read_up_to(queue, num_records=100000)
    features = tf.parse_example(serialized_example, features=FEATURES)
    image = tf.decode_raw(features['image'], tf.float32)
    image = tf.reshape(image, [-1, INPUT_SIZE, INPUT_SIZE, IMAGE_CHANNELS])
    image = tf.map_fn(lambda img:tf.image.crop_to_bounding_box(img, CENTER_OFFSET, CENTER_OFFSET, IMAGE_SIZE, IMAGE_SIZE), image)
    image = tf.map_fn(lambda img:tf.image.per_image_standardization(img), image)
    label = features['label']
    return image, tf.reshape(label, [-1])

def single_example(path):
    image_data = tf.read_file(path)
    image = tf.image.decode_png(image_data, channels=IMAGE_CHANNELS)
    image = tf.image.resize_images(image, [INPUT_SIZE, INPUT_SIZE])
    image = tf.image.crop_to_bounding_box(image, CENTER_OFFSET, CENTER_OFFSET, IMAGE_SIZE, IMAGE_SIZE)
    image = tf.image.per_image_standardization(image)
    return image
