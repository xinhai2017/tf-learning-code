import tensorflow as tf
import pickle as pk
import numpy as np
import os

CIFAR_DIR = "../cifar-10-batches-py"

def load_data(filename):
    """read data from data file."""
    with open(filename,'rb') as f:
        data = pk.load(f,encoding='bytes')
        return data[b'data'],data[b'labels']


class CifarData:
    def __init__(self,filename,need_shuffle):
        all_data = []
        all_labels = []
        for filename in filename:
            data, labels = load_data(filename)
            for data,label in zip(data,labels):
                if label in [0,1]:
                    all_data.append(data)
                    all_labels.append(label)
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1
        self._labels = np.hstack(all_labels)
        print(self._data.shape)
        print(self._labels.shape)

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self,batch_size):
        """return batch_size examples as a batch."""
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                end_indicator = batch_size
            else:
                raise Exception("have no more examples.")

        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples.")
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data,batch_labels


train_filenames = [os.path.join(CIFAR_DIR,'data_batch_%d' % i) for i in range(1,6)]
test_filenames = [os.path.join(CIFAR_DIR,'test_batch')]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)


# 搭建模型图 2nn_struct.6
x = tf.placeholder(tf.float32, [None, 3072]) #data的占位符，一个变量，
# shape y:[None]
y = tf.placeholder(tf.int64, [None]) # label的占位符，离散值故用 tf.int64     None输入数据量不定，为了使用batch_size的使用

# (3072, 1)
w = tf.get_variable('w', [x.get_shape()[-1], 1], # w的维度是和x数据的维度3072做内积的，输出是一维的
                   initializer=tf.random_normal_initializer(0, 1)) # 使用正态分布做初始化，均值为0 方差为1
# (1, )
b = tf.get_variable('b', [1],
                   initializer=tf.constant_initializer(0.0)) # 常量初始化bias

# x: [None, 3072] * w：[3072, 1] = y_: [None, 1]
y_ = tf.matmul(x, w) + b #做内积

# [None, 1]
p_y_1 = tf.nn.sigmoid(y_) # 将y变成概率值，总和为1
# reshape --> [None, 1]
y_reshaped = tf.reshape(y, (-1, 1)) #改变labels: y的shape 到概率值
y_reshaped_float = tf.cast(y_reshaped, tf.float32) # 数据类型转换
loss = tf.reduce_mean(tf.square(y_reshaped_float - p_y_1))  # 均方误差 ，计算损失函数

# bool
predict = p_y_1 > 0.5 # 预测值
# [1,0,1,1,1,0,0,0]
correct_prediction = tf.equal(tf.cast(predict, tf.int64), y_reshaped) #改变预测值为bool类型,equal返回bool值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64)) # 再把bool值转为，float64

#定义梯度下降方法
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss) #adam会调整学习率的值

    # 2nn_struct.7 执行计算图
    init = tf.global_variables_initializer()  # 初始化全局变量
    batch_size = 20  # 每个batch的数据集规模
    train_steps = 100000  # 训练步数
    test_steps = 100  # 测试步数

with tf.Session() as sess:  # 开启一个回话，执行计算图
    sess.run(init)  # 初始化全局变量
    for i in range(train_steps):  #
        batch_data, batch_labels = train_data.next_batch(batch_size)
        loss_val, acc_val, _ = sess.run(  # 调sess.run函数，执行计算图
            [loss, accuracy, train_op],  # 加 train_op的话就是在训练模型，不加为测试模型
            feed_dict={  # 要塞入的数据
                x: batch_data,
                y: batch_labels})
        if (i + 1) % 500 == 0:  # 查看中间过程
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f' % (i + 1, loss_val, acc_val))
        if (i + 1) % 5000 == 0:  #
            test_data = CifarData(test_filenames, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels = test_data.next_batch(batch_size)
                test_acc_val = sess.run(
                    [accuracy],
                    feed_dict={
                        x: test_batch_data,
                        y: test_batch_labels
                    })
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)  # 在总的数据集上求平均
            print('[Test ] Step: %d, acc: %4.5f' % (i + 1, test_acc))
