import tensorflow as tf
import visualizer
CONV1_FILTERS = 64
CONV2_FILTERS = 96
CONV3_FILTERS = 128
HIDDEN1_NODES = 384
HIDDEN2_NODES = 256
def inference(images, image_size, channels, num_classes, options={}):
    def conv(name, input_tensor, input_size, output_size, activation=tf.nn.relu):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable(name='weights', shape=[5, 5, input_size, output_size],
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(name='biases', shape=[output_size],
                dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            local = tf.nn.bias_add(tf.nn.conv2d(input_tensor, weights, [1, 1, 1, 1], padding='SAME'), biases)
            activations = activation(local)
            pool = tf.nn.max_pool(activations, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            norm = tf.nn.lrn(pool, 4)
            dropped = tf.nn.dropout(norm, 1.0 - options['dropout'])
            return dropped
    def layer(name, input_tensor, input_size, output_size, activation=tf.nn.relu):
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable(name='weights', shape=[input_size, output_size],
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            if (name != 'output'):
                weight_decay = tf.multiply(tf.nn.l2_loss(weights), options['regularization'])
                tf.add_to_collection('losses', weight_decay)
            biases = tf.get_variable(name='biases', shape=[output_size],
                dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            local = tf.matmul(input_tensor, weights) + biases
            activations = activation(local)
            return activations
    conv1 = conv('conv1', images, channels, CONV1_FILTERS)
    conv2 = conv('conv2', conv1, CONV1_FILTERS, CONV2_FILTERS)
    conv3 = conv('conv3', conv2, CONV2_FILTERS, CONV3_FILTERS)
    input_features = image_size * image_size // 64 * CONV3_FILTERS
    reshaped = tf.reshape(conv3, [-1, input_features])
    hidden1 = layer('hidden1', reshaped, input_features, HIDDEN1_NODES)
    dropped1 = tf.nn.dropout(hidden1, 1.0 - options['dropout'])
    hidden2 = layer('hidden2', dropped1, HIDDEN1_NODES, HIDDEN2_NODES)
    dropped2 = tf.nn.dropout(hidden2, 1.0 - options['dropout'])
    output = layer('output', dropped2, HIDDEN2_NODES, num_classes,
        activation=tf.identity)
    with tf.variable_scope('conv1', reuse=True) as scope:
        kernels = tf.reshape(tf.get_variable('weights'),
            [5, 5, channels, CONV1_FILTERS])
        visualization = visualizer.put_kernels_on_grid(kernels)
        tf.summary.image('kernels', visualization)
    return output

