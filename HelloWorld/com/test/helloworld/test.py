import tensorflow as tf

sess = tf.Session();

b = tf.constant([[1, 2], [3, 4]]);

size = tf.shape(b);
m = tf.slice(size, [0], [1]);
tf.
one = tf.ones(size);

print(sess.run(one));

sess.close();