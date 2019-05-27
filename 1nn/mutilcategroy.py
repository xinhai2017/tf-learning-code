import tensorflow as tf
import os
import numpy as np
import pickle as pk

CIFAR_DIR = "../cifar-10-batches-py"

def read_data(filename):
    """read data from file."""
    with open(filename,'rb') as f:
        data = pk.load(f,encoding='bytes')
    return data[b'data'],data[b'labels']

class CifarData:
    """read data, shuffle data, output batch data and labels."""
    def __init__(self,filenames,need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = read_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 -1
        self._labels = np.hstack(all_labels)
        print(self._data.shape)
        print(self._labels.shape)

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._start_indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        """shuffle data."""
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self,batch_size):
        """output next batch data and labels."""
        end_indicator = self._start_indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._start_indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples.")
        if end_indicator > self._num_examples:
            raise Exception("batch size too lager than all examples.")
        batch_data = self._data[self._start_indicator: end_indicator]
        batch_labels = self._labels[self._start_indicator: end_indicator]
        self._start_indicator = end_indicator
        return batch_data,batch_labels

train_filenames = [os.path.join(CIFAR_DIR,"data_batch_%d" % i) for i in range(1,6)]
test_filename = [os.path.join(CIFAR_DIR,'test_batch')]

train_data = CifarData(train_filenames,True)
test_data = CifarData(test_filename,False)


x = tf.placeholder(tf.float32, [None, 3072])
# [None], eg: [0,5,6,3]
y = tf.placeholder(tf.int64, [None])

# (3072, 10) 是个神经元，没有隐藏层，只有一层。
w = tf.get_variable('w', [x.get_shape()[-1], 10],
                   initializer=tf.random_normal_initializer(0, 1)) #用正态分布做初始化，0均值1方差
# (10, )
b = tf.get_variable('b', [10],
                   initializer=tf.constant_initializer(0.0)) #常数零初始化bias

# x: [None, 3072] * w: [3072, 10] = y_: [None, 10]
y_ = tf.matmul(x, w) + b

# mean square loss
"""
# course: 1 + e^x
# api: e^x / sum(e^x)
# [[0.01, 0.9, ..., 0.03], []] 分别对应十个估计值
p_y = tf.nn.softmax(y_) # 分别求指数，然后求归一化
# 5 -> [0,0,0,0,0,1,0,0,0,0]
y_one_hot = tf.one_hot(y, 10, dtype=tf.float32)
loss = tf.reduce_mean(tf.square(y_one_hot - p_y)) #平方差损失函数
"""
# cross_entropy loss
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_) #交叉熵损失函数
# y_ -> sofmax
# y -> one_hot
# loss = ylogy_



# indices
predict = tf.argmax(y_, 1)
# [1,0,1,1,1,0,0,0]
correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


init = tf.global_variables_initializer()
batch_size = 20
train_steps = 100000
test_steps = 500

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        loss_val, acc_val, _ = sess.run(
            [loss, accuracy, train_op],
            feed_dict={
                x: batch_data,
                y: batch_labels})
        if (i+1) % 500 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' \
                % (i+1, loss_val, acc_val))
        if (i+1) % 5000 == 0:
            test_data = CifarData(test_filename, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels \
                    = test_data.next_batch(batch_size)
                test_acc_val = sess.run(
                    [accuracy],
                    feed_dict = {
                        x: test_batch_data,
                        y: test_batch_labels
                    })
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Test ] Step: %d, acc: %4.5f' % (i+1, test_acc))