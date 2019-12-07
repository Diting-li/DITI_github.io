# 新程序：
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tempfile
from CNN_trainning.MNIST_format_data_reader import read_data_sets
import tensorflow as tf
dir = 'Datasets_Finished_Finally_output/Input_Data'
class CNN_model:
    def __init__(self):
        self.num_classes = 2
        self.swallowsound = read_data_sets(dir,
                                      gzip_compress=True,
                                      train_imgaes='train-audio-idx3-ubyte.gz',
                                      train_labels='train-labels-idx1-ubyte.gz',
                                      test_imgaes='test-audio-idx3-ubyte.gz',
                                      test_labels='test-labels-idx1-ubyte.gz',
                                      one_hot=True,
                                      validation_size=10,
                                      num_classes=self.num_classes,
                                      MSB=True)
    # 定义变量和卷积函数
    @staticmethod
    def conv2d(x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x5(x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 5, 1],
                              strides=[1, 2, 5, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    @staticmethod
    def weight_variable(shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    # 定义卷积层+可视化
    ############################################################
    @staticmethod
    def deepnn(self,x):
        x_image = tf.reshape(x, [-1, 13, 499, 1], name='Reshape')
            #tf.summary.image('resh_img', x_image, 10)
        # 第一层卷积层：
        W_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.tanh(self.conv2d(x_image, W_conv1) + b_conv1)
        h_conv1 = tf.layers.batch_normalization(h_conv1, training=True)
        h_pool1 = self.max_pool_2x5(h_conv1)
        #print(h_pool1.shape)
        # 第二层卷积层：
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.tanh(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_conv2 = tf.layers.batch_normalization(h_conv2, training=True)
        h_pool2 = self.max_pool_2x5(h_conv2)
        #print(h_pool2.shape)
        # 第三层卷积层：
        W_conv3 = self.weight_variable([3, 3, 64, 128])
        b_conv3 = self.bias_variable([128])
        h_conv3 = tf.nn.tanh(self.conv2d(h_pool2, W_conv3) + b_conv3)
        h_conv3 = tf.layers.batch_normalization(h_conv3, training=True)
        h_pool3 = self.max_pool_2x5(h_conv3)
        print(h_pool3.shape)
        # 第四层卷积层
        W_conv4 = self.weight_variable([2, 2, 128, 256])
        b_conv4 = self.bias_variable([256])
        h_conv4 = tf.nn.tanh(self.conv2d(h_pool3, W_conv4) + b_conv4)
        h_conv4 = tf.layers.batch_normalization(h_conv4, training=True)
        h_pool4 = self.max_pool_2x2(h_conv4)
        #print(h_pool4.shape)
        # 定义全连接层：
        # 第一层全连接层：
        W_fc1 = self.weight_variable([1 * 2 * 256, 512])
        b_fc1 = self.bias_variable([512])
        h_pool4_flat = tf.reshape(h_pool4, [-1, 1 * 2 * 256])
        h_fc1 = tf.nn.tanh(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        #print(h_fc1.shape)
        # 第二层全连接层：
        W_fc2 = self.weight_variable([512, 2])
        b_fc2 = self.bias_variable([2])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        return y_conv, keep_prob
    # Import data


    def train(self):
        # Create the model
        #with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, [None, 6487], name='x_input') #这个是图像
        # Define loss and optimizer
        y_ = tf.placeholder(tf.float32, [None, 2], name='y_input') #这个是标签
        # Build the graph for the deep net
        y_conv, keep_prob = self.deepnn(self,x)
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)
        # cross_entropy = tf.reduce_mean(cross_entropy)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        # AdamOptimizer(1e-4)
        # GradientDescentOptimizer(0.5)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        sess = tf.Session()
        # 保存tensorbord图表类
        merged =tf.summary.merge_all()
        #train_writer = tf.summary.FileWriter('logs/', sess.graph)
        sess.run(tf.global_variables_initializer())
        # 模型保存：print(ckpt)
        #with tf.Session() as sess:
            # 模型保存  step1
        saver = tf.train.Saver()
        checkpoint_dir = "./"  # 保存modle的路径
            # 返回checkpoint文件中checkpoint的状态
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            # print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:  # 如果存在以前保存的模型
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)  # 加载模型
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])  # 通过文件名得到模型保存时迭代的轮数
        else:  # 如果不存在之前保存的模型
            sess.run(tf.global_variables_initializer())  # 变量初始化
            start_step = 0
            print('start training from new state')
        print('训练开始！')

        #print(self.swallowsound.train.images.shape)
        #
        for i in range(start_step,start_step+100):
            batch = self.swallowsound.train.next_batch(10)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

            if i % 10 == 0:
                train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                a = i // 10 + 1
                print('输入第 %d 批数据, 此时的训练精度为：%g' % (a, train_accuracy))
                #result = sess.run(merged, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                #train_writer.add_summary(result, i)
                saver.save(sess, 'log/my_test_model', global_step=i)
        print("="*50)
        print('测试开始~')
        print('测试精度为：" %g' % accuracy.eval(session=sess, feed_dict={x: self.swallowsound.test.images, y_: self.swallowsound.test.labels,
                                                                   keep_prob: 0.5}) + ' "')
        #print(y_conv)
        print('测试结束！')
