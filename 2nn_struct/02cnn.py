import tensorflow as tf
import numpy as np
import os
import pickle as pk

CIFAR_DIR = "../cifar-10-batches-py"

def read_data(filename):
    """read data from file."""
    with open(filename,'rb') as f:
        data = pk.load(f,encoding='bytes')
    return data[b'data'],data[b'labels']

class CifarData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_label = []
        for filename in filenames:
            data, label = read_data(filename)
            all_data.append(data)
            all_label.append(label)
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1
        self._label = np.hstack(all_label)
        self._need_shuffle = need_shuffle
        self._indicator = 0
        self._num_examples = self._data.shape[0]

        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        """shuffle data."""
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._label = self._label[p]

    def next_batch(self,batch_size):
        """get batch size data."""
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples.")
        if end_indicator > self._num_examples:
            raise Exception("too lager batch size than examples.")
        batch_data = self._data[self._indicator: end_indicator]
        batch_label = self._label[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_label

train_filenames = [os.path.join(CIFAR_DIR,"data_batch_%d" % i) for i in range(1,6)]
test_filename = [os.path.join(CIFAR_DIR,"test_batch")]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filename, False)


x = tf.placeholder(tf.float32,[None, 3072])
y = tf.placeholder(tf.int64, [None])

x_image = tf.reshape(x,[-1,3,32,32])
x_image = tf.transpose(x_image,perm=[0,2,3,1])

conv1 = tf.layers.conv2d(x_image,
                         32,
                         (3,3),
                         padding="same",
                         activation=tf.nn.relu,
                         name="conv1"
                         )
pooling1 = tf.layers.max_pooling2d(conv1,
                                   (2,2),
                                   (2,2),
                                   name="pool1"
                                   )

conv2 = tf.layers.conv2d(pooling1,
                         32,
                         (3,3),
                         padding="same",
                         activation=tf.nn.relu,
                         name="conv2"
                         )
pooling2 = tf.layers.max_pooling2d(conv2,
                                   (2,2),
                                   (2,2),
                                   name="pool2"
                                   )
conv3 = tf.layers.conv2d(pooling2,
                         32,
                         (3,3),
                         padding="same",
                         activation=tf.nn.relu,
                         name="conv3"
                         )
pooling3 = tf.layers.max_pooling2d(conv3,
                                   (2,2),
                                   (2,2),
                                   name="pool3"
                                   )
flatten = tf.layers.flatten(pooling3)
y_ = tf.layers.dense(flatten,10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_)

predict = tf.argmax(y_,1)

correct_predict = tf.equal(predict,y)
accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float64))

with tf.name_scope("train_op"):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 100

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        train_batch_data, train_batch_label = train_data.next_batch(batch_size)
        loss_val, acc_val, _  = sess.run([loss,accuracy,train_op],feed_dict={
            x: train_batch_data,
            y: train_batch_label,
        })

        if (i+i) % 100 == 0:
            print("[Train] step: %d, acc: %4.5f, loss: %4.5f" % (i+1, acc_val, loss_val))

        if (i + 1) % 500 == 0:
            test_data = CifarData(test_filename,False)
            all_test_acc = []
            for i in range(test_steps):
                test_batch_data, test_batch_label  = test_data.next_batch(batch_size)
                test_acc_val = sess.run([accuracy],feed_dict={
                    x:test_batch_data,
                    y:test_batch_label,
                })
                all_test_acc.append(test_acc_val)
            test_acc = np.mean(all_test_acc)
            print("[Test] acc: %4.5f" % test_acc)