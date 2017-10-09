import tensorflow as tf

a = tf.constant([[1, 2], [3, 4]]);
b = tf.constant([[1, 2], [3, 4]]);

sess = tf.Session();


X = tf.constant([[1, 1], [1, 2]], tf.float64); # m x n+1
y = tf.constant([[1], [2]], tf.float64); # m x 1

m = 2;

#size = tf.shape(X);
#size = sess.run(size);
#m = size[0];
#n = size[1];

#X = tf.concat(tf.ones([2, 1], tf.int32), tf.ones([2, 1], tf.int32)) # m x n+1

param = tf.Variable(tf.zeros([2, 1], tf.float64)); # n+1 x 1

#tf.mean

#tf.matmul(X, param) - y;
#X = tf.cast(X, tf.float64);
#y = tf.cast(y, tf.float64);
h = tf.matmul(X, param);
J = tf.reduce_mean(tf.square(h-y))/2;
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(J, var_list=param)

sess.run(tf.initialize_all_variables());
#print(sess.run(tf.reduce_prod(a, reduction_indices=0)));

#print(sess.run(X*param));
for i in range(1000):
    print(sess.run([train, J]));
#print(sess.run(tf.reduce));

print(sess.run(param));

sess.close();