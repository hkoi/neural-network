import argparse
import json
import os
import sys

import tensorflow as tf
from tensorflow.contrib.metrics import confusion_matrix
from operator import itemgetter

FLAGS = None

import artclass as data
import simple as nn

def main(_):
    sess = tf.InteractiveSession()
    with tf.variable_scope('inputs') as scope:
        validation_x, validation_labels = data.validation_examples()
        validation_x_var = tf.Variable(validation_x, validate_shape=False, trainable=False, dtype=tf.float32, name='validation_images')
        validation_labels_var = tf.Variable(validation_labels, validate_shape=False, trainable=False, dtype=tf.int64, name='validation_labels')
        training_x, training_labels = data.training_examples(batch_size=FLAGS.batch_size)
        is_val = tf.placeholder(tf.bool, name='is_val')
        dropout = tf.placeholder(tf.float32)
        x = tf.cond(is_val, lambda: validation_x_var, lambda: training_x, name='images')
        labels = tf.cond(is_val, lambda: validation_labels_var, lambda: training_labels, name='labels')

    y = nn.inference(x, data.IMAGE_SIZE, data.IMAGE_CHANNELS, data.NUM_CLASSES,
            options={'dropout':dropout, 'regularization':FLAGS.regularization, 'is_val':is_val})

    with tf.variable_scope('outputs') as scope:
        y_max = tf.argmax(y, 1, name="prediction")
        corrects = tf.equal(y_max, labels, name="corrects")
        accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32), name="accuracy")
        tf.summary.scalar('accuracy', accuracy)
        # note difference between r12 and master
        confusion = confusion_matrix(labels, y_max, num_classes=data.NUM_CLASSES, name="confusion")
        y_softmax = tf.nn.softmax(y, name="y_softmax")

    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=y)
        cross_entropy_loss = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', cross_entropy_loss)
        losses = tf.get_collection('losses')
        loss = tf.add_n(losses)
        tf.summary.scalar('cross_entropy', cross_entropy_loss)
        tf.summary.scalar('total', loss)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.variable_scope('training') as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.rate)
        train = optimizer.minimize(loss, global_step=global_step)

    merged = tf.summary.merge_all()

    saver = tf.train.Saver(tf.trainable_variables() + [global_step])

    tf.train.start_queue_runners(sess=sess)
    sess.run(tf.global_variables_initializer())

    if tf.gfile.Exists(FLAGS.logdir):
        checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
        if (checkpoint):
            print('Restoring from %s' % checkpoint)
            saver.restore(sess=sess, save_path=checkpoint)
    else:
        tf.gfile.MakeDirs(FLAGS.logdir)
    
    if FLAGS.predict != '':
        print('                ', end='\t')
        for c in data.CLASSES:
            print('%7s' % c, end='\t')
        print()
        files = tf.matching_files(FLAGS.predict).eval()
        for f in files:
            image = data.single_example(f)
            image_out = sess.run(image)
            y_out = sess.run(y_softmax, feed_dict={
                dropout: 0.0, is_val: False, x: [image_out]
            })
            print('%18s' % os.path.basename(f).decode("utf-8"), end='\t')
            for i, c in enumerate(data.CLASSES):
                print('%7.2f' % (y_out[0][i] * 100.0), end='\t')
            print()
        sys.exit()

    if tf.train.global_step(sess, global_step) == 0:
        training_logger = tf.summary.FileWriter(os.path.join(FLAGS.logdir, 'training'), sess.graph)
    else:
        training_logger = tf.summary.FileWriter(os.path.join(FLAGS.logdir, 'training'))
    validation_logger = tf.summary.FileWriter(os.path.join(FLAGS.logdir, 'validation'))
    
    images_val, labels_val = sess.run([validation_x, validation_labels])
    
    for st in range(FLAGS.steps + 1):
        step = tf.train.global_step(sess, global_step)
        if step % 10 == 0:
            summary_output, training_accuracy = \
                sess.run([merged, accuracy], feed_dict={
                    dropout: 0.0, is_val: False
                })
            training_logger.add_summary(summary_output, step)
            summary_output, validation_accuracy, y_out, labels_out, y_max_out, confusion_out = \
                sess.run([merged, accuracy, y_softmax, labels, y_max, confusion], feed_dict={
                    dropout: 0.0, is_val: True
                })
            validation_logger.add_summary(summary_output, step)
            print('[Step %4d]  Training: %6.2f%%  Validation: %6.2f%%' % (step, training_accuracy * 100, validation_accuracy * 100))
            saver.save(sess, os.path.join(FLAGS.logdir, "model.ckpt"), global_step=step)

            jsondata = {
                "confusion": confusion_out.tolist(),
                "correct": round(validation_accuracy * len(labels_val)),
                "tops": [[[] for i in range(data.NUM_CLASSES)] for j in range(data.NUM_CLASSES)],
                "actuals": confusion_out.sum(1).tolist(),
                "predictions": confusion_out.sum(0).tolist(),
                "total": len(labels_val)
            }
            if step % 50 == 0:
                for i in range(len(labels_val)):
                    jsondata['tops'][int(labels_out[i])][int(y_max_out[i])].append({"idx" : i, "prob" : float(y_out[i][int(y_max_out[i])])})
                for i in range(data.NUM_CLASSES):
                    for j in range(data.NUM_CLASSES):
                        jsondata['tops'][i][j] = sorted(jsondata['tops'][i][j], key=itemgetter('prob'), reverse=True)
                with open(os.path.join("confusion", data.NAME, "summary.json"), 'w') as outfile:
                    json.dump(jsondata, outfile)
        sess.run(train, feed_dict={dropout: FLAGS.dropout, is_val: False})
        
    training_logger.close()
    validation_logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', type=str, default='', help='Predict input file')
    parser.add_argument('--batch_size', type=int, default=1024, help='No of training examples per step')
    parser.add_argument('--dropout', type=float, default=0.0, help='Drop probability for training dropout')
    parser.add_argument('--regularization', type=float, default=0.0, help='Regularization parameter')
    parser.add_argument('--rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--logdir', type=str, default='log', help='Log directory')
    parser.add_argument('--steps', type=int, default=1000, help='Steps to train')
    FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
