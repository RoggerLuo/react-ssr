import numpy as np
import tensorflow as tf

#mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# X = mnist.train.images[0:5000,:]
# y = mnist.train.labels[0:5000,:]
Xt = mnist.test.images[0:1000,:]
yt = mnist.test.labels[0:1000,:]

# 输入的
px = tf.placeholder(tf.float32,[None,784])
py = tf.placeholder(tf.float32,[None,10])
X = tf.reshape(px,[-1,28,28,1])


# 要训练的
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

# 第一层卷积
w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
conved_1 = tf.nn.relu(conv2d(X,w_conv1)+b_conv1)
pooled_1 = max_pool_2x2(conved_1)

# 第二层卷积
w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
conved_2 = tf.nn.relu(conv2d(pooled_1,w_conv2)+b_conv2)
pooled_2 = max_pool_2x2(conved_2)

# 密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(pooled_2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout层
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 最后输出10个
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# cross entropy
cost = - tf.reduce_sum(py*tf.log(y_conv))
# 训练
train_once = tf.train.AdamOptimizer(1e-4).minimize(cost)

# start
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
costList = []
for i in range(500):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_once, feed_dict = {px: batch_xs, py: batch_ys,keep_prob: 0.5})        
    costList.append(sess.run( cost, feed_dict ={px: batch_xs, py: batch_ys,keep_prob: 1}))
    if i%100 == 0:
        # 计算准确率，是拿输出的结果比对一下
        corre_boolean = tf.equal(tf.argmax(y_conv,1),tf.argmax(py,1))
        accu = tf.reduce_mean(tf.cast(corre_boolean,'float'))
        print(i)
        print(sess.run(accu, feed_dict={px: Xt, py: yt,keep_prob: 1}))

import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.plot(costList,'b-')
plt.show()
