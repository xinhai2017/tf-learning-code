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
        self._incidator = 0
        self._num_examples = self._data.shape[0]
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        """shuffle data."""
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._label = self._label[p]

    def next_batch(self,batch_size):
        """get next batch size data."""
        end_incidator = self._incidator + batch_size
        if end_incidator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._incidator = 0
                end_incidator = batch_size
            else:
                raise Exception("have no more data.")
        if end_incidator > self._num_examples:
            raise Exception("batch size too lager than example datas.")
        batch_data = self._data[self._incidator: end_incidator]
        batch_label = self._label[self._incidator: end_incidator]
        self._incidator = end_incidator
        return batch_data,batch_label

train_filenames = [os.path.join(CIFAR_DIR,"data_batch_%d" %i) for i in range(1,6)]
test_filename = [os.path.join(CIFAR_DIR,"test_batch")]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filename, False)
# batch_size = 20
# train_batch_data, train_batch_label = train_data.next_batch(batch_size)
# test_batch_data, test_batch_label = test_data.next_batch(batch_size)
# print(train_batch_label,train_batch_data)
# print(test_batch_data,test_batch_label)

def residual_block(x, output_channel):
    """residual connection implementation.
    Arg:
    - x: input image
    - output_channel: output channel number
    """
    input_channel = x.get_shape().as_list()[-1]
    if input_channel * 2 == output_channel:
        increase_dim = True
        strides = (2,2)
    elif input_channel == output_channel:
        increase_dim = False
        strides = (1,1)
    else:
        raise Exception("input channel can't match output channel.")
    conv1 = tf.layers.conv2d(x,
                             output_channel,
                             (3,3),
                             strides=strides,
                             padding="same",
                             activation=tf.nn.relu,
                             name = "conv1")
    conv2 = tf.layers.conv2d(conv1,
                             output_channel,
                             (3,3),
                             strides=(1,1),
                             padding="same",
                             activation=tf.nn.relu,
                             name="conv2")
    if increase_dim:
        pooled_x = tf.layers.average_pooling2d(x,
                                               (2,2),
                                               (2,2),
                                               padding="valid")
        padded_x = tf.pad(pooled_x,
                          [[0,0],
                           [0,0],
                           [0,0],
                           [input_channel//2,input_channel//2]])
    else:
        padded_x = x
    output_x = conv2 + padded_x
    return output_x

def rest_net(x, num_residual_blocks, num_filter_base, num_class):
    """rest net implement.
    Arg:
    - x: input image
    - num_residual_blocks: residual connection network eg:[3, 4, 6, 3]
    - num_filter_base: based output channel
    - num_class: categroy
    """
    num_subsampling = len(num_residual_blocks)
    layers = []
    input_size = x.get_shape().as_list()[1:]
    with tf.variable_scope('conv0'):
        conv0 = tf.layers.conv2d(x,
                                 num_filter_base,
                                 (3,3),
                                 strides=(1,1),
                                 padding="same",
                                 activation=tf.nn.relu,
                                 name="conv0")
        layers.append(conv0)
        for sumple_id in range(num_subsampling):
            for i in range(num_residual_blocks[sumple_id]):
                with tf.variable_scope("conv%d_%d" %(sumple_id,i)):
                    conv = residual_block(layers[-1],
                                          num_filter_base * (2 ** sumple_id))
                    layers.append(conv)
        multiplier = 2 ** (num_subsampling - 1)
        assert layers[-1].get_shape().as_list()[1:] == [input_size[0] / multiplier, input_size[1] / multiplier, num_filter_base * multiplier]
        with tf.variable_scope('fc'):
            global_pool = tf.reduce_mean(layers[-1], [1,2]) #average pooling
            logits = tf.layers.dense(global_pool, num_class)
            layers.append(logits)
        return layers[-1]

x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])

x_image = tf.reshape(x, [-1, 3, 32, 32])
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1])
logits = rest_net(x_image,
                        [3,4,6,3],
                        32,
                        10)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

predict = tf.argmax(logits,1)
corredt_predict = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(corredt_predict, tf.float32))

with tf.variable_scope("train_op"):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 100

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        train_batch_data, train_batch_label = train_data.next_batch(batch_size)
        train_loss_val, train_acc_val, _ = sess.run([loss, accuracy, train_op],feed_dict={
            x: train_batch_data,
            y: train_batch_label
        })

        if (i+1) % 100 == 0:
            print("[Train] setp: %d, loss: %4.5f, accuracy: %4.5f" % (i+1, train_loss_val, train_acc_val))

        if (i+1) % 500 == 0:
            test_data = CifarData(test_filename,False)
            test_all_acc_val = [ ]
            for i in range(test_steps):
                test_batch_data, test_batch_label = test_data.next_batch(batch_size)
                test_acc_val = sess.run([accuracy],feed_dict={
                    x:test_batch_data,
                    y:test_batch_label,
                })
                test_all_acc_val.append(test_acc_val)
            test_acc = np.mean(test_all_acc_val)
            print("[Test] accuracy: %4.5f" % test_acc)