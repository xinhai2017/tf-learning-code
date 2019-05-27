""""
1.制定面板图上显示的变量
2.训练过程中将这些变量计算出来，输出到文件中
3.文件解析 tensorboard --logdir=dir
"""
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
        self._data = self._data / 127.5 - 1
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
                           activation=tf.nn.relu,
                           name="conv1_1")
conv1_2 = tf.layers.conv2d(conv1_1,
                           32,
                           (3,3),
                           padding="same",
                           activation=tf.nn.relu,
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
                           activation=tf.nn.relu,
                           name="conv2_1")
conv2_2 = tf.layers.conv2d(conv2_1,
                           32,
                           (3,3),
                           padding="same",
                           activation=tf.nn.relu,
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
                           activation=tf.nn.relu,
                           name="conv3_1")
conv3_2 = tf.layers.conv2d(inputs=conv3_1,
                           filters=32,
                           kernel_size=(3,3),
                           strides=(1,1),
                           padding="same",
                           activation=tf.nn.relu,
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

# 功能性函数
def variable_summary(var, name):
    """constructs summary for statistics of a variable."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('mean', mean)
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.histogram('histogram', var)
with tf.name_scope('summary'):
    variable_summary(conv1_1, 'conv1_1')
    variable_summary(conv1_2, 'conv1_2')
    variable_summary(conv2_1, 'conv2_1')
    variable_summary(conv2_2, 'conv2_2')
    variable_summary(conv3_1, 'conv3_1')
    variable_summary(conv3_2, 'conv3_2')
# 1.指定显示的变量
loss_summary = tf.summary.scalar('loss',loss)
# 文件存储： key ：'loss':[(10,333),(20,34)......]
accuracy_aummary = tf.summary.scalar('accuracy',accuracy)

source_image = (x_image + 1) * 127.5
inputs_summary = tf.summary.image('inputs_image', source_image)

merged_summary = tf.summary.merge_all()
merged_summary_test = tf.summary.merge([loss_summary, accuracy_aummary])

# 2. 指定保存的文件夹
LOG_DIR = '.'
run_label = 'run_vgg_tensorboard'
run_dir = os.path.join(LOG_DIR, run_label)
if not os.path.exists(run_dir):
    os.mkdir(run_dir)
train_log_dir = os.path.join(run_dir, 'train')
test_log_dir = os.path.join(run_dir, 'test')
if not os.path.exists(train_log_dir):
    os.mkdir(train_log_dir)
if not os.path.exists(test_log_dir):
    os.mkdir(test_log_dir)

init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 100

output_summary_every_steps = 100


with tf.Session() as sess:
    sess.run(init)
    #指定两个writer,文件句柄
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    test_writer = tf.summary.FileWriter(test_log_dir)

    fixed_test_batch_data, fixed_test_batch_label = test_data.next_batch(10000)

    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        eval_ops = [loss, accuracy, train_op]
        should_output_summary = ((i+1) % output_summary_every_steps == 0)
        if should_output_summary:
            eval_ops.append(merged_summary)

        eval_ops_results = sess.run(
            eval_ops,
            feed_dict={
                x: batch_data,
                y: batch_labels})
        loss_val, acc_val = eval_ops_results[0:2]

        if should_output_summary:
            train_summary_str = eval_ops_results[-1]
            train_writer.add_summary(train_summary_str, i+1)
            test_summary_str = sess.run([merged_summary_test],feed_dict={
                                                                    x: fixed_test_batch_data,
                                                                    y: fixed_test_batch_label,
                                                                     })[0]
            test_writer.add_summary(test_summary_str, i+1)
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