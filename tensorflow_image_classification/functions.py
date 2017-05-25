import tensorflow as tf

def get_weights(n_features, n_labels):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow weights
    """
    # TODO: Return weights
    return tf.Variable(tf.truncated_normal((n_features, n_labels)))


def get_biases(n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    # TODO: Return biases
    return tf.Variable(tf.zeros(n_labels))


def linear(input, w, b):
    """
    Return linear function in TensorFlow
    :param input: TensorFlow input
    :param w: TensorFlow weights
    :param b: TensorFlow biases
    :return: TensorFlow linear function
    """
    # TODO: Linear Function (xW + b)
    return tf.add(tf.matmul(input, w), b)

# Cross entropy cost function
softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(tf.multiply(tf.constant(-1.0),tf.reduce_sum(tf.multiply(tf.log(softmax), one_hot))), feed_dict={ softmax: softmax_data, one_hot: one_hot_data})
    print output

# Divide the data into mini batches of size = batch_size
def batches(batch_size, features, labels):
    output_batches = []
    assert len(features) == len(labels)
    for start_index in range(0, len(features), batch_size):
        last_index = start_index + min(start_index + batch_size, len(features))
        batch = [features[start_index:last_index],labels[start_index:last_index]]
        output_batches.append(batch)
    return output_batches


def print_epoch_stats(cost, accuracy,epoch_i, sess, features, labels,last_features, last_labels, valid_features, valid_labels):
    """
    Print cost and validation accuracy of an epoch
    """
    current_cost = sess.run(
        cost,
        feed_dict={features: last_features, labels: last_labels})
    valid_accuracy = sess.run(
        accuracy,
        feed_dict={features: valid_features, labels: valid_labels})
    print('Epoch: {:<4} - Cost: {:<8.3} Valid Accuracy: {:<5.3}'.format(
        epoch_i,
        current_cost,
        valid_accuracy))
