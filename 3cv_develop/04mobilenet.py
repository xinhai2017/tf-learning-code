import tensorflow as tf
import pickle as pk
import os
import numpy as np

CIFAR_DIR = "../cifar-10-batches-py"


def read_file(filename):
    """read data from file."""
    with open(filename,'rb') as f:
        data = pk.load(f,encoding='bytes')
    return data[b'data'],data[b'labels']

class CifarData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_label = []
        for filename in filenames:
            data, label = read_file(filename)
            all_data.append(data)
            all_label.append(label)
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 -1
        self._label = np.hstack(all_label)
        self._need_shuffle = need_shuffle
        self._incidator = 0
        self._num_examples = self._data.shape[0]
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        """shuffle data."""
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._label = self._label[p]

    def next_batch(self, batch_size):
        """get next batch size data."""
        end_incidator = self._incidator + batch_size
        if end_incidator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._incidator = 0
                end_incidator = batch_size
            else:
                raise Exception("have no more example data.")
        if end_incidator > self._num_examples:
            raise Exception("batch size too lager than num examples")
        batch_data = self._data[self._incidator: end_incidator]
        batch_label = self._label[self._incidator: end_incidator]
        self._incidator = end_incidator
        return batch_data,batch_label

train_filenames = [os.path.join(CIFAR_DIR,"data_batch_%d" %i) for i in range(1,6)]
test_filename = [os.path.join(CIFAR_DIR,"test_batch")]

train_data = CifarData(train_filenames, True)
# test_data = CifarData(test_filename, False)

def separable_block(x, output_channel_number, name):
    """separable block implementation.
    Args:
    - x:
    - output_channel_number:output channel of 1*1 conv layer
    - name:
    """
    with tf.variable_scope(name):
        input_channel = x.get_shape().as_list()[-1]
        channel_wise_x = tf.split(x, input_channel,axis=3)
        output_channels = [ ]
        for i in range(len(channel_wise_x)):
            output_channel = tf.layers.conv2d(channel_wise_x[i],
                                              1,
                                              (3,3),
                                              strides=(1,1),
                                              padding="same",
                                              name="conv_%d" % i)
            output_channels.append(output_channel)
        concat_layer = tf.concat(output_channels,axis=3)
        conv1_1 = tf.layers.conv2d(concat_layer,
                                   output_channel_number,
                                   (1,1),
                                   strides=(1,1),
                                   padding="same",
                                   activation=tf.nn.relu,
                                   name="conv1_1")
    return conv1_1

x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])

x_image = tf.reshape(x, [-1,3,32,32])
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])

conv1 = tf.layers.conv2d(x_image,
                         32,
                         (3,3),
                         (1,1),
                         activation=tf.nn.relu,
                         padding="same",
                         name="conv1")

pooling1 = tf.layers.max_pooling2d(conv1,
                                   (2,2),
                                   (2,2),
                                   name="pooling1")

separable2a = separable_block(pooling1,32,"separable2a")
separable2b = separable_block(separable2a,32,"separable2b")

pooling2 = tf.layers.max_pooling2d(separable2b,
                                   (2,2),
                                   (2,2),
                                   name="pooling")
separable3a = separable_block(pooling2,32,name="separable3a")
separable3b = separable_block(separable3a,32,name="separable3b")

pooling3 = tf.layers.max_pooling2d(separable3b,
                                   (2,2),
                                   (2,2),
                                   name="pooling3")
flatten = tf.layers.flatten(pooling3)

logits = tf.layers.dense(flatten, 10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

predict = tf.argmax(logits,1)
correct_predict = tf.equal(predict,y)
accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))

with tf.name_scope("train_op"):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


init = tf.global_variables_initializer()
train_steps = 10000
test_steps = 100
batch_size = 20

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        train_batch_data, train_batch_label = train_data.next_batch(batch_size)
        train_loss_val, train_acc_val, _ = sess.run([loss, accuracy, train_op],
                                                    feed_dict={
                                                                x: train_batch_data,
                                                                y: train_batch_label,
                                                               })

        if (i+i) % 100 == 0:
            print("[Train] setp: %d, loss: %4.5f, acc:%4.5f" % (i+1, train_loss_val, train_acc_val))

        if (i+1) % 500 == 0:
            test_data = CifarData(test_filename, False)
            all_test_acc = []
            for i in range(test_steps):
                test_batch_data, test_batch_label = test_data.next_batch(batch_size)
                test_acc_val = sess.run([accuracy],
                                        feed_dict={
                                                    x: test_batch_data,
                                                    y: test_batch_label,
                                                  })
                all_test_acc.append(test_acc_val)
            test_acc = np.mean(all_test_acc)
            print("[Test] acc: %4.5f" % test_acc)