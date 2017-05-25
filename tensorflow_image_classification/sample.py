import tensorflow as tf

hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    output = sess.run(hello_constant)
    print(output)

    # 0-Dimensional int32 Tensor
    A = tf.constant(1234)
    # 1-Dimensional int32 Tensor
    B = tf.constant([123, 456, 789])
    # 2-Dimensional int32 Tensor
    C = tf.constant([ [123, 456, 789], [222, 333, 444] ])

    print(sess.run(A))
    print(sess.run(B))
    print(sess.run(C))


# Placeholder Basics
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(z, feed_dict={x: 'Hello World!', y: 123, z: 45.67})
    print output

# Type Conversion

with tf.Session() as sess:
    exp = tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))
    output = sess.run(exp)
    print output

# Cross entropy cost function
softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(tf.multiply(tf.constant(-1.0),tf.reduce_sum(tf.multiply(tf.log(softmax), one_hot))), feed_dict={ softmax: softmax_data, one_hot: one_hot_data})
    print output
