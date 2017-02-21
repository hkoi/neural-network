import argparse
import os
import random
import sys
import numpy as np

import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help='Dataset to process')
    parser.add_argument('trainval', type=str, help='training or validation')
    parser.add_argument('size', type=int, help='Image size')

    FLAGS, _ = parser.parse_known_args()

    sess = tf.InteractiveSession()
    image_data = tf.placeholder(dtype=tf.string)
    png_image = tf.image.decode_png(image_data, channels=3)
    png_resized = tf.image.resize_images(png_image, [FLAGS.size, FLAGS.size])
    jpeg_image = tf.image.decode_jpeg(image_data, channels=3)
    jpeg_resized = tf.image.resize_images(jpeg_image, [FLAGS.size, FLAGS.size])
    labels = []
    images = []
    with open(os.path.join(FLAGS.dataset, 'classes.txt')) as f:
      classes = [c.rstrip('\n') for c in f.readlines()]
    for index, class_name in enumerate(classes):
        path = os.path.join(FLAGS.dataset, FLAGS.trainval, class_name, '*')
        print('Processing ', path)
        file_list = tf.gfile.Glob(path)
        labels.extend([index] * len(file_list))
        for filename in file_list:
            with tf.gfile.FastGFile(filename, 'r') as f:
                raw_data = f.read()
            if ('.png' in filename) or ('.PNG' in filename):
                image = sess.run(png_resized, feed_dict={image_data: raw_data})
            else:
                image = sess.run(jpeg_resized, feed_dict={image_data: raw_data})
            images.append(image)
            print(os.path.basename(filename), end="\t", flush=True)
        print()

    n = len(labels)
    shuffler = list(range(n))
    random.seed(1000000007)
    random.shuffle(shuffler)
    images = [images[i] for i in shuffler]
    labels = [labels[i] for i in shuffler]

    with tf.python_io.TFRecordWriter(os.path.join('..', 'tensorflow', 'data', '%s_%s' % (FLAGS.dataset, FLAGS.trainval))) as writer:
        for i in range(n):
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[images[i].tostring()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]]))
            }))
            writer.write(example.SerializeToString())
    print("Wrote %d images to data/%s-%s" % (n, FLAGS.dataset, FLAGS.trainval))

    confusion = np.array(images).transpose(3, 0, 1, 2).reshape(3, -1, FLAGS.size * FLAGS.size).transpose(1, 2, 0).astype(np.uint8)
    confusion_data = tf.placeholder(dtype=tf.uint8)
    confusion_png = tf.image.encode_png(confusion_data)
    confusion_file = tf.write_file(os.path.join('..', 'tensorflow', 'confusion', FLAGS.dataset, '%s_batch_0.png' % FLAGS.trainval), confusion_png)
    sess.run([confusion_file], feed_dict = {confusion_data: confusion})
