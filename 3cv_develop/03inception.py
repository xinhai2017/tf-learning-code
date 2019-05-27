import tensorflow as tf
import pickle as pk
import os
import numpy as np

CIFAR_DIR = "../cifar-10-batches-py"

def read_file(filename):
    """read data from file."""
    with open(filename, 'rb') as f:
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
        """get next batch data."""
        end_incidator = self._indicator + batch_size
        if end_incidator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_incidator = batch_size
            else:
                raise Exception("have no more data.")

        if end_incidator > self._num_examples:
            raise Exception("batch size too lager than examples. ")
        batch_data = self._data[self._indicator: end_incidator]
        batch_label = self._label[self._indicator: end_incidator]
        self._indicator = end_incidator
        return batch_data,batch_label
train_filenames = [os.path.join(CIFAR_DIR,"data_batch_%d" %i) for i in range(1,6)]
test_filename = [os.path.join(CIFAR_DIR,"test_batch")]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filename, False)

def inception_block(x,output_channel_for_each_path,name):
    """inception block implementation.
    Arg:
    - x: input image
    - output_channel_for_each_path:
    - name
    """
    with tf.variable_scope(name):
        conv1_1 = tf.layers.conv2d(x,
                                   output_channel_for_each_path[0],
                                   kernel_size=(1,1),
                                   strides=(1,1),
                                   activation=tf.nn.relu,
                                   padding="same",
                                   name="conv1_1")
        conv3_3 = tf.layers.conv2d(x,
                                   output_channel_for_each_path[1],
                                   kernel_size=(3,3),
                                   strides=(1,1),
                                   padding="same",
                                   activation=tf.nn.relu,
                                   name="conv3_3")
        conv5_5 = tf.layers.conv2d(x,
                                   output_channel_for_each_path[2],
                                   kernel_size=(5,5),
                                   strides=(1,1),
                                   padding="same",
                                   activation=tf.nn.relu,
                                   name="conv5_5")
        max_pooling = tf.layers.max_pooling2d(x,
                                         pool_size=(2,2),
                                         strides=(2,2),
                                         name="pooled")
    max_pooling_shape = max_pooling.get_shape().as_list()[1:]
    input_shape = x.get_shape().as_list()[1:]
    padded_width = (input_shape[0] - max_pooling_shape[0]) // 2
    padded_height = (input_shape[1] - max_pooling_shape[1]) // 2
    padded_pooling = tf.pad(max_pooling,
                           [[0,0],
                            [padded_width,padded_width],
                            [padded_height, padded_height],
                            [0,0]])
    conv_layer = tf.concat([conv1_1, conv3_3, conv5_5, padded_pooling], axis=3)
    return conv_layer

x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])

x_image = tf.reshape(x, [-1,3,32,32])
x_image = tf.transpose(x_image, perm=[0,2,3,1])

conv1 = tf.layers.conv2d(x_image,
                         32,
                         (3,3),
                         (1,1),
                         padding="same",
                         activation=tf.nn.relu,
                         name="conv1")
pooling1 = tf.layers.max_pooling2d(conv1,
                                   (2,2),
                                   (2,2),
                                   name="pooling1")
inception2a = inception_block(pooling1,[32,32,32],"inpception2a")
inception2b = inception_block(inception2a,[64,64,64],"inception2b")

pooling2 = tf.layers.max_pooling2d(inception2b,
                                   (2,2),
                                   (2,2),
                                   name="pooling2")

inception3a = inception_block(pooling2,
                              [16,16,16],
                              "inception3a")
inception3b = inception_block(inception3a,
                              [16,16,16],
                              "inception3b")

pooling3 = tf.layers.max_pooling2d(inception3b,
                                   (2,2),
                                   (2,2),
                                   name="pooling3")
faltten = tf.layers.flatten(pooling3)
logits = tf.layers.dense(faltten,10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=logits)

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
        train_loss_val, train_acc_val, _ = sess.run([loss, accuracy, train_op],feed_dict={
            x: train_batch_data,
            y: train_batch_label,
        })

        if (i+1) % 100 == 0:
            print("[Train] step: %d, loss: %4.5f, acc: %4.5f" %(i+1, train_loss_val, train_acc_val))

        if (i+1) % 500 == 0:
            test_data = CifarData(test_filename,False)
            all_test_acc_val = []
            for i in range(test_steps):
                test_batch_data, test_batch_label = test_data.next_batch(batch_size)
                test_acc_val = sess.run([accuracy],feed_dict={
                    x: test_batch_data,
                    y: test_batch_label,
                })
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print("[Test] acc: %4.5f" % test_acc)