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
    def __init__(self,filenames,need_shufle):
        all_data = []
        all_label = []
        for filename in filenames:
            data, label = read_data(filename)
            all_data.append(data)
            all_label.append(label)
        self._data = np.vstack(all_data)
        self._label = np.hstack(all_label)
        print(self._data.shape)
        print(self._label.shape)

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shufle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        """shuffle data."""
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._label = self._label[p]

    def next_batch(self, batch_size):
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
            raise Exception("batch size too lager than examples.")
        batch_data = self._data[self._indicator: end_indicator]
        batch_label = self._label[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data,batch_label

train_filenames = [os.path.join(CIFAR_DIR,"data_batch_%d" %i) for i in range(1,6)]
test_filename = [os.path.join(CIFAR_DIR,"test_batch")]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filename, False)

# batch_size = 20
# train_batch_data, train_batch_label = train_data.next_batch(batch_size)
# test_batch_data, test_batch_label = test_data.next_batch(batch_size)
#
# print(train_batch_label,train_batch_data)
# print(test_batch_data,train_batch_label)

x = tf.placeholder(tf.float32,[None, 3072])
y = tf.placeholder(tf.int64, [None])

x_image = tf.reshape(x,[-1,3,32,32])
x_image = tf.transpose(x_image, perm=[0,2,3,1])
#32 x 32 x 3
conv1_1 = tf.layers.conv2d(x_image,
                           32,
                           (3,3),
                           padding="same",
                           name="conv1_1")
conv1_2 = tf.layers.conv2d(conv1_1,
                           32,
                           (3,3),
                           padding="same",
                          name="conv1_2")
#16 x 16 x 32
pooling1 = tf.layers.max_pooling2d(conv1_2,
                                   (2,2),
                                   (2,2),
                                   name="pooling1")

conv2_1 = tf.layers.conv2d(pooling1,
                           32,
                           (3,3),
                           padding="same",
                           name="conv2_1")
conv2_2 = tf.layers.conv2d(conv2_1,
                           32,
                           (3,3),
                           padding="same",
                           name="conv2_2")
#8 x 8 x 32
pooling2 = tf.layers.max_pooling2d(inputs=conv2_2,
                                   pool_size=(2,2),
                                   strides= (2,2),
                                   name="pooling2")


conv3_1 = tf.layers.conv2d(inputs=pooling2,
                           filters=32,
                           kernel_size=(3,3),
                           strides=(1,1),
                           padding="same",
                           name="conv3_1")
conv3_2 = tf.layers.conv2d(inputs=conv3_1,
                           filters=32,
                           kernel_size=(3,3),
                           strides=(1,1),
                           padding="same",
                           name="conv3_2")
#4 x 4 x 32
pooling3 = tf.layers.max_pooling2d(inputs=conv3_2,
                                   pool_size=(2,2),
                                   strides=(2,2),
                                   padding="valid",
                                   name="pooling3")
#[None, 4x4x32]
flatten = tf.layers.flatten(pooling3)
logits = tf.layers.dense(flatten,10)

loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=logits)

predict = tf.argmax(logits,1)
correct_predict = tf.equal(predict,y)
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float64))

with tf.name_scope("train_op"):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 100

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        loss_val, acc_val, _ = sess.run(
            [loss, accuracy, train_op],
            feed_dict={
                x: batch_data,
                y: batch_labels})
        if (i+1) % 100 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' \
                % (i+1, loss_val, acc_val))
        if (i+1) % 1000 == 0:
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