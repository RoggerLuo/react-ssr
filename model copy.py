import tensorflow as tf
import numpy as np
from .op_cost import get_cost
import time


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def dot(x, W):
    a = tf.transpose(tf.expand_dims(W, 1))
    b = tf.expand_dims(x, 1)
    dot = tf.matmul(a, b)
    return dot


class Paragragh(object):

    def __init__(self, config):
        self.config = config
        self.X = tf.placeholder(tf.float32, [config.input_size])
        self.Y = tf.placeholder(tf.float32, [2, ])

    def add_layer(self, input_size, output_size, activator):
        w = weight_variable([input_size, output_size])
        b = bias_variable([output_size])
        layer = activator(dot(self.X, w) + b)
        self.X = layer

    def add_dropout(self, dropout_rate):
        self.X = tf.nn.dropout(self.X, dropout_rate)

    def build(self, center, target, negs):
        # self.dropout_rate = tf.placeholder("float")

        self.add_layer(self.config.input_size, 32, tf.nn.relu)
        self.add_layer(32, 100, tf.nn.relu)
        self.add_dropout(0.3)  # self.dropout_rate
        self.add_layer(100, 2, tf.nn.relu)
        self.add_layer(16, 2, tf.nn.softmax)

    def get_cost(self):
        distant = self.Y - self.X
        cost = - tf.reduce_sum(tf.log(1 - distant))

    def train_once(self):
        train_once = tf.train.AdamOptimizer(1e-4).minimize(cost)

    def train(self):
        wv_list1
        wv_list2
        label 0, 1

        

        sess = tf.Session()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(self.config.repeate_times):
            self.sess.run(self.gd)

            # 调参数、测试性能的时候才调用
            if self.config.env == 'development':
                self.rs_list.append(self.sess.run(self.cost))
        return self

    def export(self):
        center = self.sess.run(self.center)
        target = self.sess.run(self.target)
        negs = [self.sess.run(neg) for neg in self.negs]
        return center, target, negs, self.rs_list


def test():
    class Config(object):
        repeate_times = 10

    center = np.array([1, 2, 3])
    target = np.array([4, 5, 6])
    negs = [np.array([7, 8, 9]), np.array([3, 6, 9])]

    # running test
    nn = Nn(Config)
    center, target, negs, rs_list = nn.build(
        center, target, negs).train().export()
    print(rs_list)
    print(center)
    print(target)
    print(negs)
    show(rs_list)

# test()
